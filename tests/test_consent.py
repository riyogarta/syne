import sys, time
sys.path.insert(0, "/home/syne/repo-syne")
from syne.consent import ConsentStore, make_key, content_hash

P = 0; F = 0
def ok(name, cond):
    global P, F
    if cond: P += 1; print(f"  PASS  {name}")
    else: F += 1; print(f"  FAIL  {name}")

k1 = make_key("u1","s1","x","exec","rm -rf /a")
k2 = make_key("u1","s1","x","exec","rm -rf /b")  # different content
k3 = make_key("u2","s1","x","exec","rm -rf /a")  # different user
k4 = make_key("u1","s2","x","exec","rm -rf /a")  # different session

s = ConsentStore(ttl_seconds=600)

# 1. not granted by default
ok("cold: not granted", not s.is_granted(k1))
# 2. grant then granted
s.grant(k1)
ok("after grant: granted", s.is_granted(k1))
# 3. content-bound: k2 not granted by k1
ok("content-bound: diff payload not granted", not s.is_granted(k2))
# 4. same-actor: other user not granted
ok("same-actor: diff user not granted", not s.is_granted(k3))
# 5. same-actor: other session not granted
ok("same-actor: diff session not granted", not s.is_granted(k4))
# 6. content_hash stable
ok("hash stable", content_hash("rm -rf /a") == content_hash("rm -rf /a"))
ok("hash distinct", content_hash("a") != content_hash("b"))

# 7. fixed-mode expiry
sf = ConsentStore(ttl_seconds=1, mode="fixed")
sf.grant(k1)
ok("fixed: granted immediately", sf.is_granted(k1))
time.sleep(1.1)
ok("fixed: expired after TTL", not sf.is_granted(k1))

# 8. sliding refresh keeps alive
ss = ConsentStore(ttl_seconds=2, mode="sliding")
ss.grant(k1)
time.sleep(1.2)
ok("sliding: alive mid-window (refresh)", ss.is_granted(k1))  # this refreshes
time.sleep(1.2)
ok("sliding: still alive after refresh", ss.is_granted(k1))  # would be dead if fixed

# 9. revoke
s.grant(k1)
ok("revoke removes", s.revoke(k1) and not s.is_granted(k1))
ok("revoke missing = False", not s.revoke(k1))

# 10. revoke_session
s2 = ConsentStore()
s2.grant(make_key("u1","s1","x","exec","a"))
s2.grant(make_key("u1","s1","x","exec","b"))
s2.grant(make_key("u1","s2","x","exec","c"))  # different session
n = s2.revoke_session("u1","s1")
ok("revoke_session clears matching only", n == 2 and len(s2) == 1)

# 11. sweep
s3 = ConsentStore(ttl_seconds=1, mode="fixed")
s3.grant(make_key("u1","s1","x","exec","a"))
time.sleep(1.1)
ok("sweep evicts expired", s3.sweep() == 1 and len(s3) == 0)

# 12. re-grant refreshes not duplicates
s4 = ConsentStore()
s4.grant(k1); s4.grant(k1)
ok("re-grant no duplicate", len(s4) == 1)

print(f"\n{P}/{P+F} PASS" + ("" if F==0 else f"  ({F} FAILED)"))
sys.exit(1 if F else 0)
