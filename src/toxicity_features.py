import duckdb
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
DATE = "2024-01-02"

TRADES_PATH = RAW_DIR / f"{SYMBOL}_trades_{DATE}.parquet"
BOOK_PATH = RAW_DIR / f"{SYMBOL}_book_{DATE}.parquet"
OUTPUT_PATH = PROCESSED_DIR / f"{SYMBOL}_toxicity_features_{DATE}.parquet"

# Horizon to determine toxicity in milliseconds
TOXICITY_HORIZON_MS = 500

def build_toxicity_dataset():
    print(f"Loading data for {SYMBOL} on {DATE}...")
    
    con = duckdb.connect()
    
    con.execute(f"CREATE VIEW trades AS SELECT * FROM read_parquet('{TRADES_PATH}')")
    con.execute(f"CREATE VIEW book AS SELECT * FROM read_parquet('{BOOK_PATH}')")

    print("Calculating microstructure features and toxicity target...")

    query_book = """
    WITH book_metrics AS (
        SELECT 
            transact_time,
            best_bid_price,
            best_bid_qty,
            best_ask_price,
            best_ask_qty,
            -- Mid-price and Microprice
            (best_bid_price + best_ask_price) / 2.0 AS mid_price,
            (best_bid_price * best_ask_qty + best_ask_price * best_bid_qty) / (best_bid_qty + best_ask_qty) AS microprice,
            
            -- Queue Imbalance
            (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty + 1e-9) AS queue_imbalance,
            
            -- The previous values for OFI
            LAG(best_bid_price) OVER (ORDER BY transact_time) AS prev_bid_p,
            LAG(best_bid_qty) OVER (ORDER BY transact_time) AS prev_bid_q,
            LAG(best_ask_price) OVER (ORDER BY transact_time) AS prev_ask_p,
            LAG(best_ask_qty) OVER (ORDER BY transact_time) AS prev_ask_q,
            LAG((best_bid_price + best_ask_price) / 2.0) OVER (ORDER BY transact_time) AS prev_mid
        FROM book
    ),
    book_ofi AS (
        SELECT 
            *,
            -- Order Flow Imbalance (Cont et al.)
            CASE 
                WHEN best_bid_price > prev_bid_p THEN best_bid_qty
                WHEN best_bid_price < prev_bid_p THEN -prev_bid_q
                ELSE best_bid_qty - prev_bid_q
            END 
            -
            CASE 
                WHEN best_ask_price < prev_ask_p THEN best_ask_qty
                WHEN best_ask_price > prev_ask_p THEN -prev_ask_q
                ELSE best_ask_qty - prev_ask_q
            END AS ofi,
            
            -- Deviation of microprice from mid
            (microprice - mid_price) / mid_price AS microprice_dev
        FROM book_metrics
    )
    SELECT 
        transact_time AS book_time,
        mid_price,
        microprice,
        queue_imbalance,
        ofi,
        microprice_dev
    FROM book_ofi
    WHERE ofi IS NOT NULL
    """
    
    con.execute(f"CREATE TABLE book_features AS {query_book}")

    query_join = """
    CREATE TABLE trades_with_features AS
    SELECT 
        t.transact_time,
        t.price,
        t.quantity,
        t.is_buyer_maker,
        b.mid_price AS book_mid,
        b.queue_imbalance,
        b.ofi,
        b.microprice_dev
    FROM trades t
    ASOF LEFT JOIN book_features b
    ON t.transact_time >= b.book_time
    """
    con.execute(query_join)
    
    query_target = f"""
    CREATE TABLE final_dataset AS
    WITH future_mid AS (
        SELECT 
            t.transact_time,
            t.price,
            t.quantity,
            t.is_buyer_maker,
            t.book_mid,
            t.queue_imbalance,
            t.ofi,
            t.microprice_dev,
            -- Search for mid-price in the future from book_features (+500ms)
            b.mid_price AS future_mid
        FROM trades_with_features t
        ASOF LEFT JOIN book_features b
        ON b.book_time >= (t.transact_time + {TOXICITY_HORIZON_MS})
    )
    SELECT 
        *,
        -- If agressor was buying (is_buyer_maker = False) and the price falt down -> Toxicity (1)
        -- If agressor was selling (is_buyer_maker = True) and the price raised up -> Toxicity (1)
        CASE 
            WHEN is_buyer_maker = False AND future_mid < book_mid THEN 1
            WHEN is_buyer_maker = True AND future_mid > book_mid THEN 1
            ELSE 0
        END AS is_toxic
    FROM future_mid
    WHERE future_mid IS NOT NULL
    """
    con.execute(query_target)
    
    print("Calculating VPIN and exporting...")
    
    con.execute(f"""
        COPY (
            SELECT 
                *,
                -- Simplier VPIN
                AVG(ABS(quantity * CASE WHEN is_buyer_maker = False THEN 1 ELSE -1 END)) 
                OVER (ORDER BY transact_time ROWS BETWEEN 1000 PRECEDING AND CURRENT ROW) 
                / NULLIF(AVG(quantity) OVER (ORDER BY transact_time ROWS BETWEEN 1000 PRECEDING AND CURRENT ROW), 0) 
                AS vpin_approx
            FROM final_dataset
        ) TO '{OUTPUT_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY);
    """)
    
    df_check = con.execute(f"SELECT COUNT(*) as rows, SUM(is_toxic) as toxic_trades FROM read_parquet('{OUTPUT_PATH}')").fetchdf()
    print(f"\n[SUCCESS] Saved dataset with {df_check['rows'].iloc[0]:,} trades.")
    print(f"Toxic trades: {df_check['toxic_trades'].iloc[0]:,} ({df_check['toxic_trades'].iloc[0]/df_check['rows'].iloc[0]*100:.2f}%)")
    
    con.close()

if __name__ == "__main__":
    if not TRADES_PATH.exists() or not BOOK_PATH.exists():
        print("ERROR: Raw data missing. Run data_loader.py first.")
    else:
        build_toxicity_dataset()