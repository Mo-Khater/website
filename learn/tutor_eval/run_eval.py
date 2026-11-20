#!/usr/bin/env python3
"""
Run the AI tutor evaluation set against a running learn worker.

It sets up a dedicated eval course through the public API (once; reused via
the state file), indexes it, then asks every case under every policy and
scores the answers (see scoring.py and README.md).

  python3 tutor_eval/run_eval.py --base-url http://127.0.0.1:8787 \\
      --jwt-secret "$JWT_SECRET" --host-id eval-host --student-id eval-student

Standard library only. Each full run makes about 80 chat-model calls.
"""

import argparse
import base64
import hashlib
import hmac
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring  # noqa: E402

HERE = Path(__file__).resolve().parent


def mint_token(user_id: str, secret: str, role: str = "member", ttl: int = 6 * 3600) -> str:
    """Same format as worker.create_token (HMAC-SHA256 signed, base64 JSON payload)."""
    now = int(time.time())
    payload = base64.b64encode(
        json.dumps(
            {"id": user_id, "username": user_id, "role": role, "iat": now, "exp": now + ttl}, separators=(",", ":")
        ).encode()
    ).decode()
    return payload + "." + hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


class Client:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def call(self, method, path, token=None, body=None, timeout=180):
        req = urllib.request.Request(
            self.base + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            except ValueError:
                return exc.code, {"error": f"HTTP {exc.code}"}


def need(status, body, what):
    if status != 200:
        sys.exit(f"{what} failed ({status}): {body.get('error') or body}")
    return body.get("data")


def setup_course(client, dataset, host, student, state_path):
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("dataset_version") == dataset["version"] and state.get("activity_id"):
        print(f"Using eval course {state['activity_id']} from {state_path}")
        return state["activity_id"]

    course = dataset["course"]
    print("Creating the eval course…")
    act = need(
        *client.call("POST", "/api/activities", host, {"title": course["title"], "description": course["description"]}),
        "create activity",
    )["id"]
    need(
        *client.call(
            "PATCH",
            f"/api/activities/{act}",
            host,
            {"learning_objectives": course["learning_objectives"], "prerequisites": course["prerequisites"]},
        ),
        "set objectives",
    )
    for i, session in enumerate(course["sessions"], start=1):
        need(
            *client.call(
                "POST",
                "/api/sessions",
                host,
                {
                    "activity_id": act,
                    "title": session["title"],
                    "description": session["description"],
                    "start_time": f"2026-01-{i:02d}T10:00",
                    "end_time": f"2026-01-{i:02d}T11:00",
                },
            ),
            "create session",
        )
    need(*client.call("POST", "/api/join", student, {"activity_id": act}), "student join")
    state_path.write_text(json.dumps({"dataset_version": dataset["version"], "activity_id": act}, indent=1))
    return act


def wait_until_searchable(client, act, student, case, timeout=180):
    """Vectorize is eventually consistent: wait until a known question finds its lesson."""
    deadline = time.time() + timeout
    query = urllib.parse.urlencode({"q": case["question"], "k": 3})
    while time.time() < deadline:
        status, body = client.call("GET", f"/api/activities/{act}/tutor/search?{query}", student)
        hits = (body.get("data") or {}).get("results") or []
        if status == 200 and any(h["source_title"] == case["expect_source"] for h in hits):
            return True
        time.sleep(5)
    return False


def top_retrieval_scores(client, act, student, cases):
    scores = {}
    for case in cases:
        query = urllib.parse.urlencode({"q": case["question"], "k": 1})
        status, body = client.call("GET", f"/api/activities/{act}/tutor/search?{query}", student)
        hits = ((body.get("data") or {}).get("results") or []) if status == 200 else []
        scores[case["id"]] = hits[0]["score"] if hits else None
    return scores


def ask(client, act, student, case):
    """Ask any history turns first (same conversation), then the scored question."""
    conversation = None
    for turn in case.get("history", []):
        status, body = client.call(
            "POST",
            f"/api/activities/{act}/tutor/chat",
            student,
            {"message": turn, **({"conversation_id": conversation} if conversation else {})},
        )
        if status == 200:
            conversation = body["data"]["conversation_id"]
    started = time.time()
    status, body = client.call(
        "POST",
        f"/api/activities/{act}/tutor/chat",
        student,
        {"message": case["question"], **({"conversation_id": conversation} if conversation else {})},
    )
    elapsed = round(time.time() - started, 2)
    if status != 200:
        return {"answer": "", "citations": [], "outcome": f"error {status}: {body.get('error')}"}, elapsed
    return body["data"], elapsed


def print_report(summary, thresholds, latencies):
    for policy, entry in summary.items():
        o = entry["overall"]
        print(f"\n=== policy: {policy}   overall {o['passed']}/{o['total']} ({o['rate']:.0%})")
        for kind, r in entry["by_kind"].items():
            print(f"  {kind:14} {r['passed']:>3}/{r['total']:<3} {r['rate']:.0%}")
        checks = ", ".join(
            f"{k} {v['passed']}/{v['total']}" for k, v in entry["answerable_checks"].items() if v["total"]
        )
        print(f"  answerable checks: {checks}")
        for f in entry["failures"]:
            print(f"    ✗ {f['case']}: {'; '.join(f['why'])}")
    if latencies:
        print(f"\nlatency: median {statistics.median(latencies):.1f}s, max {max(latencies):.1f}s")
    if thresholds:
        t = thresholds
        print(
            f"\nretrieval floor (MIN_RELEVANCE): answerable top scores "
            f"{t['answerable']['min']}–{t['answerable']['max']}"
            f" (median {t['answerable']['median']}), out-of-scope {t['out_of_scope']['min']}–{t['out_of_scope']['max']}"
            f" (median {t['out_of_scope']['median']})"
        )
        print(
            f"  current {t['current_floor']}: keeps {t['current']['answerable_kept']} answerable, "
            f"blocks {t['current']['out_of_scope_blocked']} out-of-scope before the model"
        )
        print(
            f"  best    {t['best_floor']}: keeps {t['best']['answerable_kept']} answerable, "
            f"blocks {t['best']['out_of_scope_blocked']} out-of-scope"
            f"{' (cleanly separable)' if t['separable'] else ' (ranges overlap)'}"
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8787")
    p.add_argument("--jwt-secret", help="JWT_SECRET of the deployment, to sign tokens for the two users")
    p.add_argument("--host-id", help="existing user id that will host the eval course")
    p.add_argument("--student-id", help="existing user id that will join it as a student")
    p.add_argument("--host-token", help="instead of --jwt-secret/--host-id")
    p.add_argument("--student-token", help="instead of --jwt-secret/--student-id")
    p.add_argument("--dataset", default=str(HERE / "dataset.json"))
    p.add_argument("--state", default=".tutor-eval-state.json", help="remembers the eval course between runs")
    p.add_argument("--policies", help="comma-separated subset of the dataset's policies")
    p.add_argument("--cases", help="comma-separated subset of case ids")
    p.add_argument("--min-relevance", type=float, default=0.5, help="the deployed tutor_chat.MIN_RELEVANCE")
    p.add_argument("--out", default="tutor-eval-report.json")
    args = p.parse_args(argv)

    if args.jwt_secret and args.host_id and args.student_id:
        host, student = mint_token(args.host_id, args.jwt_secret), mint_token(args.student_id, args.jwt_secret)
    elif args.host_token and args.student_token:
        host, student = args.host_token, args.student_token
    else:
        p.error("give --jwt-secret with --host-id and --student-id, or --host-token and --student-token")

    dataset = json.loads(Path(args.dataset).read_text())
    cases = dataset["cases"]
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    policies = dataset["policies"]
    if args.policies:
        policies = {k: v for k, v in policies.items() if k in args.policies.split(",")}

    client = Client(args.base_url)
    act = setup_course(client, dataset, host, student, Path(args.state))
    need(*client.call("PUT", f"/api/activities/{act}/tutor/policy", host, next(iter(policies.values()))), "set policy")
    stats = need(*client.call("POST", f"/api/activities/{act}/tutor/reindex", host), "reindex")
    print(f"Indexed: {stats['chunks']} chunks ({stats['embedded']} embedded now)")
    probe = next(c for c in dataset["cases"] if c["kind"] == "answerable")
    if stats["embedded"] and not wait_until_searchable(client, act, student, probe):
        sys.exit("New content never became searchable; is Vectorize bound?")

    print("Measuring retrieval scores…")
    scores = top_retrieval_scores(client, act, student, cases)
    thresholds = scoring.threshold_analysis(scores, cases, args.min_relevance)

    results, latencies = [], []
    for name, policy in policies.items():
        need(*client.call("PUT", f"/api/activities/{act}/tutor/policy", host, policy), f"set policy {name}")
        print(f"\nPolicy {name}: asking {len(cases)} questions…")
        for case in cases:
            response, elapsed = ask(client, act, student, case)
            latencies.append(elapsed)
            score = scoring.score_case(case, response, policy["mode"])
            print(f"  {'✓' if score['pass'] else '✗'} {case['id']:24} {response.get('outcome')}")
            results.append(
                {
                    "policy": name,
                    "mode": policy["mode"],
                    "case": case,
                    "score": score,
                    "response": response,
                    "seconds": elapsed,
                    "top_retrieval_score": scores.get(case["id"]),
                }
            )

    summary = scoring.summarize(results)
    print_report(summary, thresholds, latencies)
    Path(args.out).write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "dataset_version": dataset["version"],
                "activity_id": act,
                "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "summary": summary,
                "retrieval_threshold": thresholds,
                "results": results,
            },
            indent=1,
        )
    )
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
