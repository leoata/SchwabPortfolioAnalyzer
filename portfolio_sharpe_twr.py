# portfolio_sharpe_twr.py
# pip install schwab-py pandas numpy python-dateutil
import os, sys, math, json, argparse, itertools
from datetime import datetime, timedelta, timezone, date
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import schwab

from schwab.auth import easy_client
from schwab.client import Client
import dotenv

dotenv.load_dotenv()
# --- Config ---
EQUITY_ASSET_TYPES = {"EQUITY", "ETF", "MUTUAL_FUND"}  # price history supported per docs
TRD_TYPES_INCLUDE = {"TRADE"}
DIV_TYPES_INCLUDE = {"DIVIDEND_OR_INTEREST"}  # count as return, not external flow
CASHFLOW_TYPES = {  # exclude from return via TWR adjustment
    "ACH_RECEIPT", "ACH_DISBURSEMENT", "CASH_RECEIPT", "CASH_DISBURSEMENT",
    "ELECTRONIC_FUND", "WIRE_IN", "WIRE_OUT", "JOURNAL", "MARGIN_CALL", "MONEY_MARKET", "RECEIVE_AND_DELIVER",
    "SMA_ADJUSTMENT", "MEMORANDUM"
}


def build_client():
    key = os.environ.get("SCHWAB_API_KEY");
    secret = os.environ.get("SCHWAB_APP_SECRET")
    cb = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
    tok = os.environ.get("SCHWAB_TOKEN_PATH", "./schwab_token.json")
    if not key or not secret:
        print("Set SCHWAB_API_KEY and SCHWAB_APP_SECRET.", file=sys.stderr);
        sys.exit(1)
    return easy_client(api_key=key, app_secret=secret, callback_url=cb, token_path=tok)


def get_account_hashes(c: Client):
    r = c.get_account_numbers();
    r.raise_for_status()
    return [x["hashValue"] for x in r.json()]


def get_equity_positions_now(c: Client) -> pd.Series:
    """Return current equity/ETF quantity per symbol across all accounts."""
    r = c.get_accounts(fields=[Client.Account.Fields.POSITIONS])
    r.raise_for_status()
    qty = {}
    for acct in r.json() or []:
        for p in (acct.get("securitiesAccount", {}) or {}).get("positions", []) or []:
            inst = p.get("instrument", {}) or {}
            at = (inst.get("assetType") or "").upper()
            t = (inst.get("type") or "").upper()
            is_equity = at == "EQUITY"
            is_etf = at == "COLLECTIVE_INVESTMENT" and t == "EXCHANGE_TRADED_FUND"
            if not (is_equity or is_etf):
                continue
            sym = inst.get("symbol")
            if not sym:
                continue
            q = float(p.get("longQuantity") or 0.0) - float(p.get("shortQuantity") or 0.0)
            qty[sym] = qty.get(sym, 0.0) + q

    return pd.Series(qty, dtype=float)


# helper
def group_percentages(qty: pd.DataFrame, px: pd.DataFrame, groups: dict, port_val: pd.Series) -> pd.DataFrame:
    """
    qty, px share the same index (dates) and columns (symbols).
    groups: {"Label": ["SYM1","SYM2",...], ...}
    Returns DataFrame of % of total portfolio value per group per day.
    """
    # value by symbol
    val = (qty.reindex(columns=px.columns, fill_value=0.0) * px).fillna(0.0)

    out = {}
    cols_available = set(val.columns)
    for label, syms in groups.items():
        keep = [s for s in syms if s in cols_available]
        if not keep:
            # still include empty series to keep legend stable
            out[label] = pd.Series(0.0, index=val.index)
            continue
        out[label] = val[keep].sum(axis=1)

    grp_val = pd.DataFrame(out).sort_index()
    pct = (grp_val.div(port_val.replace(0, np.nan), axis=0) * 100.0).fillna(0.0)
    return pct


def plot_group_percentages(pct_df: pd.DataFrame, title: str = "Portfolio allocation by group (%)",
                           outfile: str | None = None):
    plt.figure(figsize=(12, 6))
    for col in pct_df.columns:
        plt.plot(pct_df.index, pct_df[col], label=col)
    plt.legend(loc="best")
    plt.ylabel("% of portfolio value")
    plt.xlabel("Date")
    plt.title(title)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=144)
    else:
        plt.show()


def windowed(start_dt: datetime, end_dt: datetime, days=60):
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=days), end_dt)
        yield cur, nxt
        cur = nxt


def pull_transactions(c: Client, account_hash: str, start: datetime, end: datetime):
    # Iterate 60-day windows per Schwab API constraint. :contentReference[oaicite:1]{index=1}
    all_tx = []
    for s, e in windowed(start, end, days=60):
        rr = c.get_transactions(account_hash, start_date=s, end_date=e)
        rr.raise_for_status()
        all_tx.extend(rr.json() or [])
    return all_tx


def parse_trade(tx):
    """
    Schwab 'TRADE' payloads carry legs in tx['transferItems'].
    Keep only security legs (EQUITY or ETF under COLLECTIVE_INVESTMENT).
    Ignore currency/fees. Return list[(symbol, dqty, trade_date)].
    """
    import pandas as pd

    trade_ts = tx.get("tradeDate") or tx.get("time") or tx.get("transactionDate")
    d = pd.to_datetime(trade_ts, utc=True).date() if trade_ts else None

    out = []
    for leg in tx.get("transferItems", []) or []:
        # Skip fee or currency legs
        if "feeType" in leg:
            continue
        inst = leg.get("instrument") or {}
        sym = inst.get("symbol")
        if not sym or sym.startswith("CURRENCY_"):
            continue

        asset_type = (inst.get("assetType") or "").upper()
        inst_type = (inst.get("type") or "").upper()
        is_equity = asset_type == "EQUITY"
        is_etf = asset_type == "COLLECTIVE_INVESTMENT" and inst_type == "EXCHANGE_TRADED_FUND"
        if not (is_equity or is_etf):
            continue

        # Quantity sign is in 'amount': buys positive, sells negative in examples.
        amt = leg.get("amount")
        if amt is None:
            amt = leg.get("quantity", 0)
        qty = float(amt or 0.0)

        # Safety: if broker encodes positive qty with CLOSING, flip sign.
        pe = (leg.get("positionEffect") or "").upper()
        if pe == "CLOSING" and qty > 0:
            qty = -qty

        if qty != 0 and d is not None:
            out.append((sym, qty, d))
    return out


def collect_activity(c: Client, acct_hashes, lookback_days: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days + 5)  # buffer
    trades, dividends, cashflows = [], [], []
    for ah in acct_hashes:
        txs = pull_transactions(c, ah, start, end)
        for tx in txs:
            ttype = (tx.get("type") or "").upper()
            if ttype in TRD_TYPES_INCLUDE:
                rows = parse_trade(tx)
                for sym, dqty, dte in rows:
                    trades.append((dte, sym, dqty, ah))
            elif ttype in DIV_TYPES_INCLUDE:
                amt = 0
                for ti in (tx.get("transferItems") or []):
                    amt += float(ti.get("amount") or 0.0)
                dte = pd.to_datetime(tx.get("tradeDate") or tx.get("settlementDate")).date()
                # treat dividend as return cash on that date
                dividends.append((dte, amt))
            elif ttype in CASHFLOW_TYPES:
                # if tx.get("type") == "CASH_RECEIPT", its a deposit from my bank
                amt = float(tx.get("netAmount") or tx.get("amount") or 0.0)
                dte = pd.to_datetime(tx.get("tradeDate") or tx.get("settlementDate")).date()
                cashflows.append((dte, amt))
    tr_df = pd.DataFrame(trades, columns=["date", "symbol", "dqty", "account_hash"]) if trades else pd.DataFrame(
        columns=["date", "symbol", "dqty", "account_hash"])
    dv_df = pd.DataFrame(dividends, columns=["date", "amount"]) if dividends else pd.DataFrame(
        columns=["date", "amount"])
    cf_df = pd.DataFrame(cashflows, columns=["date", "amount"]) if cashflows else pd.DataFrame(
        columns=["date", "amount"])
    return tr_df, dv_df, cf_df


def get_price_panel(c: Client, symbols, start_dt: datetime, end_dt: datetime):
    frames = []
    for sym in sorted(set(symbols)):
        rr = c.get_price_history_every_day(sym, start_datetime=start_dt, end_datetime=end_dt,
                                           need_extended_hours_data=False, need_previous_close=False)
        if rr.status_code != 200: continue
        js = rr.json() or {}
        cds = js.get("candles") or []
        if not cds: continue
        df = pd.DataFrame(cds)
        df["date"] = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_convert("UTC").dt.date
        frames.append(df[["date", "close"]].rename(columns={"close": sym}).drop_duplicates("date").set_index("date"))
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def rebuild_quantity_matrix(trades_df: pd.DataFrame, date_index: pd.Index, baseline_qty: pd.Series | None = None):
    """
    SOMETHIGN IS WRONG HERE MAKING THE qty COME OUT WITH NaN VALUES WHICH FUCKS UP PORT_VAL
    :param trades_df:
    :param date_index:
    :param baseline_qty:
    :return:
    """
    if trades_df.empty and baseline_qty is None:
        return pd.DataFrame(index=date_index)

    # Ensure date dtype matches date_index (python date, not datetime64)
    if not trades_df.empty:
        trades_df = trades_df.assign(date=pd.to_datetime(trades_df["date"]).dt.date)

    # Fill missing symbol–date cells with 0, then align to full date index
    pivot = (
        trades_df.pivot_table(
            index="date", columns="symbol", values="dqty",
            aggfunc="sum", fill_value=0.0  # <- key fix
        ).reindex(date_index, fill_value=0.0)
        if not trades_df.empty else pd.DataFrame(0.0, index=date_index, columns=[])
    )
    pivot = pivot.astype(float)  # optional, keeps math clean

    qty = pivot.cumsum()  # safe because pivot has no NaNs

    if baseline_qty is not None:
        # union of columns, then add baseline shares
        qty = qty.reindex(columns=sorted(set(qty.columns) | set(baseline_qty.index)), fill_value=0.0)
        qty = qty.add(baseline_qty.reindex(qty.columns).fillna(0.0), axis="columns")
    return qty


def show_equity_portfolio_percentage_plot(qty, px, port_val, args):
    # get top 10 holdings by % allocation at the end date excluding anything with above 20% allocation
    equities_to_show = {}
    if port_val.empty:
        return
    latest_date = port_val.index[-1]
    latest_val_by_sym = (qty.loc[latest_date] * px.loc[latest_date
    ]).fillna(0.0)
    latest_pct_by_sym = (latest_val_by_sym / port_val.loc[latest_date]).fillna(1.0) * 100.0
    top_syms = latest_pct_by_sym[latest_pct_by_sym < 20.0].sort_values(ascending=False).head(10).index.tolist()
    for sym in top_syms:
        equities_to_show[sym] = [sym]

    pct_df = group_percentages(qty, px, equities_to_show, port_val)
    # Optional CLI to save the plot
    if getattr(args, "plot", "") == "":
        plot_group_percentages(pct_df, outfile=args.plot)
    else:
        # Print a compact JSON preview for verification
        preview = pct_df.tail(5).round(2).reset_index().rename(columns={"index": "date"})
        print(json.dumps({"allocation_pct_last5": preview.reset_index().to_dict(orient="records")}, indent=2,
                         default=str))


def compute_twr_daily(port_val, ext_flows):
    # ext_flows: external contributions/withdrawals only (exclude dividends)
    ext = ext_flows.reindex(port_val.index).fillna(0.0)
    pv = port_val
    r = (pv.diff() - ext) / pv.shift(1)
    return r.dropna()


def sharpe_from_daily(excess_daily):
    mu = excess_daily.mean()
    sd = excess_daily.std(ddof=1)
    return math.sqrt(252) * (mu / sd) if sd > 0 else float("nan")


# since the transactions api doesnt include info about how many shares left after a transaction, impossible to reconstruct baseline from txns alone
# So, we combine today's positions along with all of the info we can gather about the state of the account at the lookback start date
def build_baseline(c, acct_hashes, trades_df):
    baseline_qty = {}
    for ah in acct_hashes:
        # first, fill baseline_qty with current positions
        account = c.get_account(ah, fields=schwab.client.Client.Account.Fields.POSITIONS).json()
        positions = account['securitiesAccount']['positions']
        for position in filter(lambda p: 'instrument' in p and p['instrument']['assetType'] == "EQUITY" or \
                                         p['instrument']['assetType'] == "COLLECTIVE_INVESTMENT" and p['instrument'][
                                             'type'] == "EXCHANGE_TRADED_FUND"
                , positions):
            netQuantity = position['longQuantity'] - position['shortQuantity']
            symbol = position['instrument']['symbol']
            if symbol not in baseline_qty:
                baseline_qty[symbol] = netQuantity

        # Next, use the trades_df to reverse engineer original baseline
        account_trades = trades_df.copy()
        # filter trades to only those from this account
        account_trades = account_trades[trades_df['account_hash'] == ah]
        # group by symbol and sum dqty to get net change per symbol
        net_changes = account_trades.groupby('symbol')['dqty'].sum()
        for symbol, net_change in net_changes.items():
            if symbol in baseline_qty:
                baseline_qty[symbol] -= net_change
            else:
                baseline_qty[symbol] = -net_change
    return pd.Series(baseline_qty, dtype=float)


def main():
    ap = argparse.ArgumentParser(
        description="Compute Sharpe ratio using Schwab transactions (position changes handled).")
    ap.add_argument("--lookback", type=int, default=365,
                    help="Calendar days back to scan transactions and prices. Default 252.")
    ap.add_argument("--rf", type=float, default=0.02, help="Annual risk-free (e.g., 0.02 = 2%).")
    ap.add_argument("--plot", type=str, default="", help="Path to save PNG of allocation lines")
    args = ap.parse_args()
    pd.set_option('display.max_rows', None)

    c = build_client()
    acct_hashes = get_account_hashes(c)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.lookback + 5)

    # --- after collecting activity ---
    trades_df, div_df, cash_df = collect_activity(c, acct_hashes, lookback_days=args.lookback + 5)
    # 1) Baseline shares at the lookback start: positions_now - net_changes_in_window
    positions_now = get_equity_positions_now(c)  # equities + ETFs only
    net_changes = (trades_df.groupby("symbol")["dqty"].sum() if not trades_df.empty else pd.Series(dtype=float))
    # fill in missing symbols with 0 change
    net_changes = net_changes.reindex(positions_now.index, fill_value=0.0)

    baseline_qty = build_baseline(c, acct_hashes, trades_df)

    # 2) Build the symbol universe for PRICES = baseline ∪ traded
    symbols_for_px = sorted(
        set(baseline_qty.index.tolist()) | set(trades_df["symbol"].tolist() if not trades_df.empty else []))
    if not symbols_for_px:
        print(json.dumps({"error": "No equity symbols to value."}, indent=2));
        sys.exit(0)

    # 3) Download prices for the full universe so baseline quantities are revalued daily
    px = get_price_panel(c, symbols_for_px, start_dt, end_dt)
    if px.empty:
        print(json.dumps({"error": "No price history for symbols."}, indent=2));
        sys.exit(1)

    dates = px.index
    # 4) Rebuild daily quantities = baseline_at_start + cumulative trade deltas
    qty = rebuild_quantity_matrix(trades_df, dates, baseline_qty=baseline_qty.reindex(px.columns).fillna(0.0))
    qty = qty.reindex(columns=px.columns, fill_value=0.0)
    # 5) Value portfolio using the full price panel (captures price moves of baseline holdings)
    val_by_sym = (qty * px)
    port_val = val_by_sym.sum(axis=1)

    # External flows per day (exclude dividends). Sum by date.
    ext_flows = cash_df.groupby("date")["amount"].sum() if not cash_df.empty else pd.Series(dtype=float)
    ext_flows = ext_flows.reindex(dates).fillna(0.0)

    # Include dividends as part of return: add to portfolio value change on dividend date.
    if not div_df.empty:
        div_by_day = div_df.groupby("date")["amount"].sum().reindex(dates).fillna(0.0)
        # Add dividend cash to value series so it contributes to returns
        port_val_adj = port_val + div_by_day.cumsum()
    else:
        div_by_day = pd.Series(0.0, index=dates)
        port_val_adj = port_val

    daily_ret_twr = compute_twr_daily(port_val_adj, ext_flows)

    rf_daily = (1 + args.rf) ** (1 / 252) - 1
    sharpe = sharpe_from_daily(daily_ret_twr - rf_daily)

    out = {
        "symbols": symbols_for_px,
        "n_trading_days": int(len(daily_ret_twr)),
        "mean_daily_return": float(daily_ret_twr.mean()),
        "std_daily_return": float(daily_ret_twr.std(ddof=1)),
        "annual_rf": float(args.rf),
        "sharpe_annualized": float(sharpe),
    }

    print(json.dumps(out, indent=2))
    show_equity_portfolio_percentage_plot(qty, px, port_val, args)


if __name__ == "__main__":
    main()
