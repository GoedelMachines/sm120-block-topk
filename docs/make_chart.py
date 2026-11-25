# Renders block-topk.png / .svg from the numbers in ../results/bench_block_topk.log and ../results/graph_mem.log.
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as tk
S="#fcfcfb"; T1="#0b0b0b"; T2="#52514e"; GRID="#e4e3df"; DENSE="#2a78d6"; BLOCK="#eb6834"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.edgecolor":GRID,"text.color":T1,
  "axes.labelcolor":T2,"xtick.color":T2,"ytick.color":T2})
fig,(a,b)=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={"width_ratios":[2.1,1]},facecolor=S)
kinds=["Candidate consumers\n(layers 24–36)","Candidate source\n(layer 20)","Plain layers\n(2, 8, 14)"]
dense=[11538,13672,5385]; block=[210,7706,3640]   # us, 512 rows x 1M context
h=0.34
for i,(d,k) in enumerate(zip(dense,block)):
    y=len(kinds)-1-i
    a.barh(y+h/2+0.02,d,h,color=DENSE,label="Dense (vLLM-Moet sm_120 path)" if i==0 else None)
    a.barh(y-h/2-0.02,k,h,color=BLOCK,label="Block top-k (this repo)" if i==0 else None)
    a.text(d+150,y+h/2+0.02,f"{d/1000:.1f} ms",va="center",fontsize=10,color=T2)
    sp=d/k
    lab=f"{k/1000:.2f} ms  ({sp:.0f}× faster)" if sp>=10 else f"{k/1000:.1f} ms  ({sp:.1f}× faster)"
    a.text(k+150,y-h/2-0.02,lab,va="center",fontsize=10,color=T1)
a.set_yticks(range(len(kinds))); a.set_yticklabels(kinds[::-1])
a.set_xlabel("Time per indexer call, 512 rows × 1M-token context (CUDA-graph replay)")
a.set_xlim(0,17500); a.xaxis.set_major_formatter(tk.FuncFormatter(lambda v,_: f"{v/1000:.0f} ms"))
a.grid(axis="x",color=GRID,lw=0.8); a.set_axisbelow(True)
for s in ["top","right","left"]: a.spines[s].set_visible(False)
a.tick_params(axis="y",length=0)
a.set_title("Faster per call, bit-identical output",loc="left",fontsize=12,color=T1,pad=10)
a.legend(loc="lower right",frameon=False,fontsize=10)
b.bar([0],[6.000],0.55,color=DENSE); b.bar([1],[0.570],0.55,color=BLOCK)
b.text(0,6.12,"6.00 MiB",ha="center",fontsize=10,color=T2)
b.text(1,0.69,"0.57 MiB\n(10.5× less)",ha="center",fontsize=10,color=T1)
b.set_xticks([0,1]); b.set_xticklabels(["Dense","Block top-k"])
b.set_ylabel("CUDA-graph memory per decode row"); b.set_ylim(0,7)
b.yaxis.set_major_formatter(tk.FuncFormatter(lambda v,_: f"{v:.0f} MiB"))
b.grid(axis="y",color=GRID,lw=0.8); b.set_axisbelow(True)
for s in ["top","right","bottom"]: b.spines[s].set_visible(False)
b.tick_params(axis="x",length=0)
b.set_title("Smaller graph memory",loc="left",fontsize=12,color=T1,pad=10)
for ax in (a,b): ax.set_facecolor(S)
fig.suptitle("DeepSeek-V4.1-Flash sparse indexer on RTX PRO 6000 Blackwell (sm_120): dense vs exact block top-k",
             x=0.01,ha="left",fontsize=13,fontweight="bold",color=T1)
fig.tight_layout(rect=(0,0,1,0.93))
fig.savefig("block-topk.png",dpi=160,facecolor=S); fig.savefig("block-topk.svg",facecolor=S)
