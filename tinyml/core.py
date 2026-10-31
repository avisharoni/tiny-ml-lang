"""A tiny ML-style language: let, fun, if, ints, bools, application, + - * < ==.

Pipeline: tokenize -> recursive-descent parse -> Hindley-Milner type inference
(Algorithm W with let-polymorphism) -> environment-based evaluation with closures.
"""
import re
from dataclasses import dataclass

TOK = re.compile(r"\s*(?:(\d+)|(->|==|[()+\-*<=\\])|([A-Za-z_]\w*))")
KW = {"let", "rec", "in", "if", "then", "else", "fun", "true", "false"}

def tokenize(s):
    out, i = [], 0
    s = s.rstrip()
    while i < len(s):
        m = TOK.match(s, i)
        if not m: raise SyntaxError(f"bad char at {i}: {s[i]!r}")
        out.append(int(m.group(1)) if m.group(1) is not None else (m.group(2) or m.group(3)))
        i = m.end()
    return out

# AST as tuples
class P:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def eat(self, x=None):
        v = self.peek()
        if x is not None and v != x: raise SyntaxError(f"expected {x}, got {v}")
        self.i += 1; return v
    def expr(self):
        v = self.peek()
        if v == "let":
            self.eat(); rec = self.peek() == "rec"
            if rec: self.eat()
            name = self.eat(); params = []
            while self.peek() != "=": params.append(self.eat())
            self.eat("="); body = self.expr()
            for p in reversed(params): body = ("fun", p, body)
            self.eat("in"); return ("letrec" if rec else "let", name, body, self.expr())
        if v == "fun":
            self.eat(); p = self.eat(); self.eat("->"); return ("fun", p, self.expr())
        if v == "if":
            self.eat(); c = self.expr(); self.eat("then"); a = self.expr(); self.eat("else")
            return ("if", c, a, self.expr())
        return self.cmp()
    def cmp(self):
        l = self.add()
        while self.peek() in ("<", "=="): op = self.eat(); l = ("op", op, l, self.add())
        return l
    def add(self):
        l = self.mul()
        while self.peek() in ("+", "-"): op = self.eat(); l = ("op", op, l, self.mul())
        return l
    def mul(self):
        l = self.app()
        while self.peek() == "*": self.eat(); l = ("op", "*", l, self.app())
        return l
    def app(self):
        f = self.atom()
        while self.peek() is not None and (isinstance(self.peek(), int) or self.peek() == "(" or (isinstance(self.peek(), str) and self.peek()[0].isalpha() and self.peek() not in KW - {"true", "false"})):
            f = ("app", f, self.atom())
        return f
    def atom(self):
        v = self.eat()
        if isinstance(v, int): return ("int", v)
        if v in ("true", "false"): return ("bool", v == "true")
        if v == "(": e = self.expr(); self.eat(")"); return e
        if isinstance(v, str) and v[0].isalpha(): return ("var", v)
        raise SyntaxError(f"unexpected {v}")

def parse(s):
    p = P(tokenize(s)); e = p.expr()
    if p.peek() is not None: raise SyntaxError(f"trailing {p.peek()}")
    return e

# ---- types ----
@dataclass(eq=False)
class TVar:
    id: int; ref: object = None
def TCon(name, *args): return ("T", name, args)
INT, BOOL = TCon("int"), TCon("bool")
def arrow(a, b): return TCon("->", a, b)
class TypeError_(Exception): pass

def prune(t):
    while isinstance(t, TVar) and t.ref is not None: t = t.ref
    return t
def occurs(v, t):
    t = prune(t)
    return t is v if isinstance(t, TVar) else any(occurs(v, a) for a in t[2])
def unify(a, b):
    a, b = prune(a), prune(b)
    if isinstance(a, TVar):
        if a is not b:
            if occurs(a, b): raise TypeError_("infinite type")
            a.ref = b
    elif isinstance(b, TVar): unify(b, a)
    elif a[1] != b[1] or len(a[2]) != len(b[2]): raise TypeError_(f"cannot unify {show(a)} with {show(b)}")
    else:
        for x, y in zip(a[2], b[2]): unify(x, y)

class Inf:
    def __init__(self): self.n = 0
    def fresh(self): self.n += 1; return TVar(self.n)
    def ftv(self, t):
        t = prune(t)
        return {t} if isinstance(t, TVar) else set().union(*[self.ftv(a) for a in t[2]]) if t[2] else set()
    def inst(self, sch):
        qs, t = sch; m = {q: self.fresh() for q in qs}
        def go(t):
            t = prune(t)
            if isinstance(t, TVar): return m.get(t, t)
            return ("T", t[1], tuple(go(a) for a in t[2]))
        return go(t)
    def gen(self, env, t):
        envv = set().union(*[self.ftv(s[1]) - set(s[0]) for s in env.values()]) if env else set()
        return (list(self.ftv(t) - envv), t)
    def w(self, env, e):
        k = e[0]
        if k == "int": return INT
        if k == "bool": return BOOL
        if k == "var":
            if e[1] not in env: raise TypeError_(f"unbound {e[1]}")
            return self.inst(env[e[1]])
        if k == "fun":
            a = self.fresh(); return arrow(a, self.w({**env, e[1]: ([], a)}, e[2]))
        if k == "app":
            f = self.w(env, e[1]); x = self.w(env, e[2]); r = self.fresh(); unify(f, arrow(x, r)); return r
        if k == "if":
            unify(self.w(env, e[1]), BOOL); a = self.w(env, e[2]); unify(a, self.w(env, e[3])); return a
        if k == "op":
            unify(self.w(env, e[2]), INT); unify(self.w(env, e[3]), INT)
            return BOOL if e[1] in ("<", "==") else INT
        if k == "let":
            t = self.w(env, e[2]); return self.w({**env, e[1]: self.gen(env, t)}, e[3])
        if k == "letrec":
            a = self.fresh(); t = self.w({**env, e[1]: ([], a)}, e[2]); unify(a, t)
            return self.w({**env, e[1]: self.gen(env, t)}, e[3])
        raise ValueError(k)

def show(t, names=None):
    names = {} if names is None else names; t = prune(t)
    if isinstance(t, TVar):
        if t not in names: names[t] = "'" + "abcdefghijklmnopqrstuvwxyz"[len(names) % 26]
        return names[t]
    if t[1] == "->":
        a = show(t[2][0], names); a = f"({a})" if "->" in a else a
        return f"{a} -> {show(t[2][1], names)}"
    return t[1]

def infer(src): return show(Inf().w({}, parse(src)))

# ---- evaluation ----
class Closure:
    def __init__(self, p, body, env): self.p, self.body, self.env = p, body, env

OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "<": lambda a, b: a < b, "==": lambda a, b: a == b}

def evaluate(e, env=None):
    env = env or {}
    while True:  # loop for tail calls in if/let/app
        k = e[0]
        if k in ("int", "bool"): return e[1]
        if k == "var": return env[e[1]]
        if k == "fun": return Closure(e[1], e[2], env)
        if k == "op": return OPS[e[1]](evaluate(e[2], env), evaluate(e[3], env))
        if k == "if": e = e[2] if evaluate(e[1], env) else e[3]; continue
        if k == "let": env = {**env, e[1]: evaluate(e[2], env)}; e = e[3]; continue
        if k == "letrec":
            env = dict(env); c = evaluate(e[2], env); env[e[1]] = c
            if isinstance(c, Closure): c.env = env
            e = e[3]; continue
        if k == "app":
            f = evaluate(e[1], env); x = evaluate(e[2], env)
            env = {**f.env, f.p: x}; e = f.body; continue
        raise ValueError(k)

def run(src):
    t = infer(src); return evaluate(parse(src)), t
