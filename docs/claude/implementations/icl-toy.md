# ICL toy (y=|x*g|)
`examples/baselines/ppo/icl_toy.py` — pure-torch CPU MLP, no env. Train x>0 only, test x<0.
- BC: MLP(x,g)->x*g. ICL: MLP(x, g*m, m, y', c)->y*, y'=x*g' (fresh random g' each step), c=y*-y', g dropped (m=0) w.p. p; w.p. 0.2 context zeroed so pass@1 input (y'=0,c=0) is in-dist.
- pass@1: y1=pi(x,g,0,0); pass@2: y2=pi(x,g,y1,y*-y1). BC y2=y1. Sign of c is y*-y' everywhere (spec's pass@2 text had the opposite sign).
- Run: `~/venvs/maniskill/bin/python examples/baselines/ppo/icl_toy.py` (~2 min, 5 sizes x 6 methods x 5 seeds). Output: logs/toy/{results.json,toy.png}.
