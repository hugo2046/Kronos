"""离线复核冻结产物、因果日期和账户；不调用模型或行情接口。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from loguru import logger

OUT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    """计算文件摘要。

    :param path: 输入文件。
    :return: SHA256。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path: Path) -> dict:
    """读取JSON。

    :param path: 输入文件。
    :return: JSON对象。
    """
    return json.loads(path.read_text())

def wide(path: Path) -> pd.DataFrame:
    """统一信号方向。

    :param path: 信号文件。
    :return: 日期乘股票矩阵。
    """
    f = pd.read_parquet(path)
    if not isinstance(f.index, pd.DatetimeIndex):
        f = f.T
    f.index = pd.to_datetime(f.index)
    return f.sort_index().sort_index(axis=1)

p = read(OUT/'protocol.json')
gate = read(OUT/'review_gate.json')
result = read(OUT/'result.json')
protocol_sha = hashlib.sha256(json.dumps(p, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
assert protocol_sha == gate['protocol_sha'] and gate['approved']
assert digest(OUT/'evidence_migration.json') == gate['evidence_migration_sha256']
assert digest(OUT/'root_reference_parity.json') == gate['root_reference_parity_sha256']
audit = dict(metrics={}, accounts={}, intervals={}, transactions={}, prediction={}, requests={}, historical_incident_retained=True)
daily_ics = []
for w, (n, cells) in p['expected'].items():
    refs = {}
    for tag in ['g142','state42','state100']:
        path = OUT/'verification/references'/f'{w}_{tag}.parquet'
        assert digest(path) == p['reference_files'][p['reference_paths'][w][tag]]
        refs[tag] = wide(path)
    baseline = refs['g142']
    candidate = wide(OUT/'signals'/f'{w}.parquet')
    assert candidate.index.equals(baseline.index) and candidate.columns.equals(baseline.columns)
    np.testing.assert_array_equal(np.isfinite(candidate), np.isfinite(baseline))
    assert len(candidate) == n and np.isfinite(candidate).to_numpy().sum() == cells
    sm = read(OUT/'signals'/f'{w}.json')
    assert sm['sha256'] == digest(OUT/'signals'/f'{w}.parquet')
    assert sm['source_hashes'] == gate['source_hashes'] and sm['protocol_sha'] == protocol_sha
    signals = {'原G1': baseline, '状态适配MC40': (refs['state42']+refs['state100'])*.5, 'MC40-TTA2': candidate}
    dates = candidate.index.strftime('%Y-%m-%d').tolist()
    previous = ''; before = [1.,1.]; valid = 0; skip = 0; snapshots = {}; updates = []
    for i, date in enumerate(dates):
        d = OUT/'replay'/w/'days'/date
        commit = read(d/'commit.json')
        assert commit['identity'] == {'protocol_sha':protocol_sha,'source_hashes':gate['source_hashes']}
        assert commit['previous'] == previous
        for name, expected in commit['files'].items():
            assert digest(d/name) == expected
        previous = digest(d/'commit.json')
        u = read(d/'update.json'); updates.append(u)
        assert u['date'] == date and u['a_before'] == before
        assert np.all((np.array(u['a']) >= .75) & (np.array(u['a']) <= 1.25))
        snap = pd.read_pickle(d/'snapshot.pkl'); snapshots[date] = snap
        assert snap['date'] == date and snap['issued_a'] == u['a']
        for entry in snap['entries'].values():
            assert len(entry['dates']) == 90 and max(entry['dates']) <= pd.Timestamp(date)
        s42 = pd.read_pickle(d/'signal_42.pkl'); s100 = pd.read_pickle(d/'signal_100.pkl')
        pd.testing.assert_series_equal(((s42+s100)*.5).sort_index(),candidate.loc[pd.Timestamp(date)].dropna().sort_index(),check_names=False,check_exact=True)
        if i < 11:
            assert not u['origin'] and u['a'] == [1.,1.] and u['valid'] == 0
        else:
            assert u['origin'] == dates[i-11] and u['label_end'] == dates[i-1]
            labels = pd.read_pickle(d/'labels.pkl')
            assert len(labels) == u['valid']
            for code, label in labels.items():
                expected_dates = pd.to_datetime(dates[i-10:i])
                np.testing.assert_array_equal(pd.to_datetime(label['dates']), expected_dates)
                origin_raw = snapshots[u['origin']]['entries'][code]['raw']
                np.testing.assert_allclose(label['return10'], label['raw'][-1,3]/origin_raw[-1,3]-1, atol=1e-12,rtol=0)
            target = pd.Series({code:label['return10'] for code,label in labels.items()})
            row = {'window':w,'origin':u['origin'],'mature_date':date,'samples':len(target)}
            for name, matrix in signals.items():
                issued = matrix.loc[pd.Timestamp(u['origin'])].reindex(target.index)
                assert np.isfinite(issued).all()
                row[name] = issued.corr(target,method='spearman')
            daily_ics.append(row)
        valid += u['valid']; skip += len(u['skipped']); before = u['a']
    diag = result[w]['diagnostics']
    assert valid == diag['mature_samples'] and skip == diag['skipped'] and n-11 == diag['updates']
    audit['transactions'][w] = dict(days=n,finite_cells=cells,updates=n-11,mature_samples=valid,skipped=skip,final_a=before,last_commit_sha=previous)
    ic = pd.DataFrame([x for x in daily_ics if x['window']==w])
    audit['prediction'][w] = {name:float(ic[name].mean()) for name in signals}
    np.testing.assert_allclose(ic['MC40-TTA2'].mean(),diag['mature_rankic'],atol=1e-12,rtol=0)
    nets = {}
    for name, tag in [('原G1','G1_original'),('状态适配MC40','B'),('MC40-TTA2','MC40-TTA2')]:
        folder = OUT/('accounts' if tag=='MC40-TTA2' else 'root_reference_accounts')/w/tag
        f = pd.read_parquet(folder/'report.parquet')
        assert f.index.equals(candidate.index)
        x = (f['return']-f.cost).to_numpy(); nets[name] = x
        nav = np.r_[1.,np.cumprod(1+x)]
        m = dict(net_sum=x.sum(),compound=nav[-1]-1,sharpe=np.sqrt(252)*x.mean()/x.std(ddof=1),mdd=np.min(nav/np.maximum.accumulate(nav)-1),cost=f.cost.sum(),turnover=f.turnover.mean())
        np.testing.assert_allclose(list(m.values()),[result[w]['metrics'][name][key] for key in m],atol=1e-12,rtol=0)
        audit['metrics'][w+'/'+name] = m
        if tag != 'MC40-TTA2': continue
        meta = read(folder/'metadata.json')
        assert meta['signal_sha256']==sm['sha256'] and meta['protocol_sha']==protocol_sha and meta['source_hashes']==gate['source_hashes']
        assert meta['account']['reconciliation']['exact'] and meta['account']['execution_timing']['verified']
        for fname, expected in meta['artifact_hashes'].items(): assert digest(folder/fname)==expected
        pos=pd.read_parquet(folder/'positions.parquet').set_index(['date','instrument'])
        tx=pd.read_parquet(folder/'trades.parquet').set_index(['date','instrument'])
        bal=pd.read_parquet(folder/'balances.parquet').set_index('date')
        t=tx.join(pos[['end_price']]);t['trade_pnl']=t.quantity*(t.end_price-t.price);t['cash_used']=t.quantity*t.price+t.cost
        sums=t.groupby(level=[0,1])[['quantity','trade_pnl','cost']].sum().reindex(pos.index).fillna(0)
        np.testing.assert_allclose(pos.end_amount-pos.start_amount,sums.quantity,atol=1e-8,rtol=0)
        pnl=pos.start_amount*(pos.end_price-pos.start_price)+sums.trade_pnl
        stock=pd.read_parquet(folder/'stock_daily_contributions.parquet').set_index(['date','instrument']).reindex(pos.index)
        np.testing.assert_allclose(pnl/pos.previous_nav,stock.gross_contribution,atol=1e-12,rtol=0)
        np.testing.assert_allclose((pnl-sums.cost)/pos.previous_nav,stock.net_contribution,atol=1e-12,rtol=0)
        daily=(pnl/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0)
        cost=(sums.cost/pos.previous_nav).groupby('date').sum().reindex(f.index,fill_value=0)
        np.testing.assert_allclose(daily,f['return'],atol=1e-12,rtol=0);np.testing.assert_allclose(cost,f.cost,atol=1e-12,rtol=0)
        cash=bal.start_cash-t.groupby('date').cash_used.sum().reindex(bal.index,fill_value=0)
        equity=(pos.end_amount*pos.end_price).groupby('date').sum().reindex(bal.index,fill_value=0)
        np.testing.assert_allclose(cash,bal.end_cash,atol=1e-6,rtol=0);np.testing.assert_allclose(equity+bal.end_cash,bal.nav,atol=1e-6,rtol=0)
        audit['accounts'][w] = dict(max_daily_error=float(abs(daily-f['return']).max()),trades=len(tx),positions=len(pos))
    starts=np.random.default_rng(100).integers(0,n,(5000,(n+19)//20))
    indices=((starts[:,:,None]+np.arange(20))%n).reshape(5000,-1)[:,:n]
    for name in ['原G1','状态适配MC40']:
        diff=nets['MC40-TTA2']-nets[name]
        q=np.quantile(diff[indices].sum(axis=1),[.025,.975]);record=result[w]['comparisons'][name]
        np.testing.assert_allclose([diff.sum(),*q],[record['observed_sum'],record['lower_95'],record['upper_95']],atol=1e-12,rtol=0)
        audit['intervals'][w+'/'+name] = dict(delta=float(diff.sum()),ci=q.tolist())
for path in [OUT/'market_requests.jsonl', OUT/'evaluation_requests.jsonl', OUT/'root_reference_requests.jsonl', *(OUT/'accounts').rglob('market_requests.jsonl')]:
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    ends=[]
    for row in rows:
        end=row.get('effective_end',row.get('end'))
        if end is not None:
            end=pd.Timestamp(end);assert end<=pd.Timestamp('2026-07-24');ends.append(end)
            if 'as_of' in row: assert end<=pd.Timestamp(row['as_of'])
        if 'sql' in row: assert row.get('cache')=='bypassed'
    audit['requests'][str(path.relative_to(OUT))] = dict(rows=len(rows),max_end=str(max(ends)) if ends else None)
pd.DataFrame(daily_ics).to_csv(OUT/'verification/paired_rankic.csv',index=False)
audit['passed']=True
(OUT/'verification/local_final_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
logger.info('独立复算通过：{}',audit['prediction'])
