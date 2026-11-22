import requests
import zipfile
import io
import os
import duckdb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
DATE = "2024-01-02" 

def process_csv_with_duckdb(csv_path, parquet_path, table_schema):
    con = duckdb.connect()
    
    schema_str = ", ".join([f"'{k}': '{v}'" for k, v in table_schema.items()])
    
    query = f"""
        COPY (
            SELECT * FROM read_csv('{csv_path}', 
                header = true, 
                columns = {{{schema_str}}},
                strict_mode = false,
                delim = ','
            )
        ) TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION SNAPPY);
    """
    
    con.execute(query)
    con.close()
    
    if os.path.exists(csv_path):
        os.remove(csv_path)

def download_binance_data(symbol, date_str):
    """
    Скачивает aggTrades и bookTicker с data.binance.vision (Futures).
    """
    base_url = f"https://data.binance.vision/data/futures/um/daily"
    
    trades_url = f"{base_url}/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    print(f"Downloading Trades: {trades_url}")
    trades_res = requests.get(trades_url)
    
    if trades_res.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(trades_res.content)) as z:
            csv_name = z.namelist()[0]
            temp_csv_path = RAW_DIR / f"temp_{csv_name}"
            
            with open(temp_csv_path, 'wb') as f:
                f.write(z.read(csv_name))
                
            parquet_path = RAW_DIR / f"{symbol}_trades_{date_str}.parquet"
            
            schema = {
                'agg_trade_id': 'BIGINT',
                'price': 'DOUBLE',
                'quantity': 'DOUBLE',
                'first_trade_id': 'BIGINT',
                'last_trade_id': 'BIGINT',
                'transact_time': 'BIGINT',
                'is_buyer_maker': 'BOOLEAN'
            }
            
            print("  Processing with DuckDB...")
            process_csv_with_duckdb(temp_csv_path, parquet_path, schema)
            print(f"  [SUCCESS] Saved to {parquet_path.name}")
    else:
        print(f"  [ERROR] Failed to download trades: HTTP {trades_res.status_code}")

    book_url = f"{base_url}/bookTicker/{symbol}/{symbol}-bookTicker-{date_str}.zip"
    print(f"\nDownloading BookTicker: {book_url}")
    book_res = requests.get(book_url)
    
    if book_res.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(book_res.content)) as z:
            csv_name = z.namelist()[0]
            temp_csv_path = RAW_DIR / f"temp_{csv_name}"
            
            with open(temp_csv_path, 'wb') as f:
                f.write(z.read(csv_name))
                
            parquet_path = RAW_DIR / f"{symbol}_book_{date_str}.parquet"
            
            schema = {
                'update_id': 'BIGINT',
                'best_bid_price': 'DOUBLE',
                'best_bid_qty': 'DOUBLE',
                'best_ask_price': 'DOUBLE',
                'best_ask_qty': 'DOUBLE',
                'transact_time': 'BIGINT',
                'event_time': 'BIGINT'
            }
            
            print("  Processing with DuckDB...")
            process_csv_with_duckdb(temp_csv_path, parquet_path, schema)
            print(f"  [SUCCESS] Saved to {parquet_path.name}")
    else:
        print(f"  [ERROR] Failed to download bookTicker: HTTP {book_res.status_code}")

if __name__ == "__main__":
    print(f"=== Starting Data Ingestion for {SYMBOL} on {DATE} ===\n")
    download_binance_data(SYMBOL, DATE)
    print("\n=== Ingestion Complete ===")