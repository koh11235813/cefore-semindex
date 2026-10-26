"""Per-p summaries of deliveries.csv and bytes.csv -> summary_h1.csv, summary_h2.csv (+ stdout)."""
import csv, statistics as st
from collections import defaultdict
D = list(csv.DictReader(open("deliveries.csv")))
C = ["down_pre","poll_wait_after_up","poll_wait_noflap","pre_other_up","poll_rtt","fetch_down","fetch_up"]
g = defaultdict(list)
for r in D: g[r["p"]].append(r)
rows=[]; base=None
for p in sorted(g):
    rs=g[p]; tot=[float(r["total"]) for r in rs]; mt=st.mean(tot)
    row={"p":p,"n":len(rs),"total_mean_ms":round(mt),"total_median_ms":round(st.median(tot)),
         "frac_down_at_upstream_store":round(sum(int(r["down_at_up"]) for r in rs)/len(rs),3),
         "frac_with_fetch_timeout":round(sum(int(r["n_fetch_timeout"])>0 for r in rs)/len(rs),3)}
    for c in C:
        v=[float(r[c]) for r in rs]; row[c+"_mean"]=round(st.mean(v)); row[c+"_median"]=round(st.median(v)); row[c+"_share"]=round(st.mean(v)/mt,3)
    rows.append(row)
b=rows[0]
for row in rows:
    dt=row["total_mean_ms"]-b["total_mean_ms"]; row["delta_total_ms"]=dt
    for c in C:
        d=row[c+"_mean"]-b[c+"_mean"]; row["delta_"+c+"_ms"]=d; row["delta_share_"+c]=round(d/dt,3) if dt else ""
# poll wait combined (after_up + noflap) and its delta
for row in rows:
    row["poll_wait_any_mean"]=row["poll_wait_after_up_mean"]+row["poll_wait_noflap_mean"]
w=csv.DictWriter(open("summary_h1.csv","w",newline=""),fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for row in rows:
    print(row["p"],"n",row["n"],"tot mean/med",row["total_mean_ms"],row["total_median_ms"],"dT",row["delta_total_ms"])
    for c in C: print("   %-20s mean %6d med %6d share %.3f  delta %6s (%s)"%(c,row[c+"_mean"],row[c+"_median"],row[c+"_share"],row["delta_"+c+"_ms"],row["delta_share_"+c]))
    print("   poll_wait_any mean", row["poll_wait_any_mean"], "down_at_up", row["frac_down_at_upstream_store"], "fetch_timeout frac", row["frac_with_fetch_timeout"])
# conditional poll-wait means
for p in sorted(g):
    au=[float(r["poll_wait_after_up"]) for r in g[p] if float(r["poll_wait_after_up"])>0 or float(r["down_pre"])>0]
    nf=[float(r["poll_wait_noflap"]) for r in g[p] if float(r["down_pre"])==0 and float(r["poll_wait_after_up"])==0]
    print("p",p,"flap-affected n",len(au),"mean wait after up",round(st.mean(au)) if au else None,"| no-flap n",len(nf),"mean wait",round(st.mean(nf)))
# H2
B=list(csv.DictReader(open("bytes.csv")))
K=["state_req","fetch_int","state_rep","fetch_data"]
runs=defaultdict(lambda: defaultdict(float))
for r in B:
    k=(r["p"],r["seed"]); R=runs[k]; R["ncr"]=float(r["n_records_created"])
    for t in K: R[t+"_B"]+=float(r[t+"_B"]); R[t+"_n"]+=float(r[t+"_n"])
    R["logged"]+=float(r["logged_sent"]); R["recon"]+=float(r["recon_sent"])
out=[]
for p in sorted({k[0] for k in runs}):
    rs=[v for k,v in runs.items() if k[0]==p]
    row={"p":p,"runs":len(rs),"records_per_run":st.mean(v["ncr"] for v in rs)}
    for t in K:
        row[t+"_B_per_rec"]=round(st.mean(v[t+"_B"]/v["ncr"] for v in rs),1)
        row[t+"_n_per_run"]=round(st.mean(v[t+"_n"] for v in rs),1)
    row["recon_B_per_rec"]=round(st.mean(v["recon"]/v["ncr"] for v in rs),1)
    row["logged_B_per_rec"]=round(st.mean(v["logged"]/v["ncr"] for v in rs),1)
    row["resid_frac"]=round(st.mean((v["logged"]-v["recon"])/v["logged"] for v in rs),4)
    out.append(row)
w=csv.DictWriter(open("summary_h2.csv","w",newline=""),fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
for r in out: print(r)
