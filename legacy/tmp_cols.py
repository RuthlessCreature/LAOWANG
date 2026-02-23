import argparse
import ui
ns=argparse.Namespace(config='config.ini',db_url=None,db=None)
eng=ui.make_engine(ui.resolve_db_target(ns))
app=ui.AppContext(eng,min_trade_date=None,job_runner=None)
cache=app._relay_cache_data()
df=cache['pred_pool']
print('rows',len(df))
print('cols',len(df.columns))
print(','.join(df.columns))
