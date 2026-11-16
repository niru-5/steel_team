#!/usr/bin/env python
"""Minimal MCP stdio client used to smoke-test `python -m wrc_demo.apps.mcp_server`
in mock mode (added by the hackathon setup; NOT upstream). Spawns the server with
WRC_ARM=mock, WRC_VIEW=0, WRC_STREAM=0, then: initialize -> tools/list ->
world_state -> list_objects -> camera_snapshot -> pick_and_place(red cube) -> move_home.
The mock arm never touches hardware. Pass --no-motion to skip the motion calls."""
import json, os, subprocess, sys, time

def main():
    no_motion = "--no-motion" in sys.argv
    env = dict(os.environ, WRC_ARM="mock", WRC_VIEW="0", WRC_STREAM="0", YOLO_OFFLINE="True",
               PYTHONNOUSERSITE="1", WRC_RUN_DIR=os.environ.get("WRC_RUN_DIR", "runs/mcp_smoke"))
    srv = subprocess.Popen([sys.executable, "-m", "wrc_demo.apps.mcp_server"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=open("setup_logs/mcp_server_stderr.log", "w"),
                           text=True, env=env)
    nid = 0
    def rpc(method, params=None, notify=False):
        nonlocal nid
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            nid += 1; msg["id"] = nid
        srv.stdin.write(json.dumps(msg) + "\n"); srv.stdin.flush()
        if notify:
            return None
        line = srv.stdout.readline()
        if not line:
            raise RuntimeError("server closed stdout")
        return json.loads(line)
    try:
        r = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "wrc-smoke", "version": "0"}})
        print("initialize ->", json.dumps(r["result"]["serverInfo"]), "protocol", r["result"]["protocolVersion"])
        rpc("notifications/initialized", notify=True)
        tools = rpc("tools/list")["result"]["tools"]
        print(f"tools/list -> {len(tools)} tools:")
        for t in tools:
            print(f"  - {t['name']}: {t['description'][:90]}")
        calls = [("world_state", {}), ("list_objects", {}), ("camera_snapshot", {})]
        if not no_motion:
            calls += [("pick_and_place", {"object": "red cube"}), ("move_home", {})]
        time.sleep(2)  # let perception pre-warm make a detector pass
        for name, args in calls:
            t0 = time.time()
            r = rpc("tools/call", {"name": name, "arguments": args})
            res = r.get("result", {})
            parts = []
            for c in res.get("content", []):
                parts.append(c["text"][:300] if c.get("type") == "text" else f"<{c.get('type')} {c.get('mimeType', '')} {len(c.get('data', ''))} b64 chars>")
            print(f"tools/call {name} ({time.time() - t0:.1f}s) isError={res.get('isError')} -> {' | '.join(parts)}")
        print("ping ->", rpc("ping").get("result"))
    finally:
        srv.stdin.close()
        try:
            srv.wait(timeout=20)
        except subprocess.TimeoutExpired:
            srv.terminate()
        print("server exit code:", srv.returncode)

if __name__ == "__main__":
    main()
