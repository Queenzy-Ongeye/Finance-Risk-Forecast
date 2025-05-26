import pandas as pd
import numpy as np

def load_and_prepare(filepath):
    """
    Load a CSV file containing stock price data and prepare it for analysis.
    Specifically designed to handle the format shown in the provided image.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    DataFrame with price and log_return columns indexed by date
    """
    try:
        # Looking at the image, the file seems to have multi-level headers
        # First try to read with skiprows to handle the format seen in the image
        df = pd.read_csv(filepath, skiprows=[0, 1])
        
        # Check if the first column is a date column
        if 'Date' not in df.columns and len(df.columns) > 0:
            # Rename the first column to 'Date' as it appears to contain dates
            df = df.rename(columns={df.columns[0]: 'Date'})
        
        # If that didn't work, try different approaches
        if 'Date' not in df.columns:
            # Try reading with header=None and then setting column names
            df = pd.read_csv(filepath, header=None)
            
            # Check if row 2 (index 2) contains "Date"
            if len(df) > 2 and "Date" in str(df.iloc[2, 0]):
                # Skip the first 3 rows and use the 4th row as data
                df = pd.read_csv(filepath, skiprows=3)
                df = df.rename(columns={df.columns[0]: 'Date'})
            else:
                # Try reading with different header configurations
                for header_row in range(0, 5):  # Try different header positions
                    try:
                        df = pd.read_csv(filepath, header=header_row)
                        if 'Date' in df.columns or any('date' in col.lower() for col in df.columns):
                            break
                    except:
                        continue
                
                # Rename date column if found with different case
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                if date_cols:
                    df = df.rename(columns={date_cols[0]: 'Date'})
                else:
                    # If nothing else works, assume the first column contains dates
                    df = df.rename(columns={df.columns[0]: 'Date'})
    
        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Find the close price column
        close_cols = [col for col in df.columns if 'close' in col.lower()]
        
        if close_cols:
            close_col = close_cols[0]
        elif 'Close' in df.columns:
            close_col = 'Close'
        elif len(df.columns) > 1:  # If no obvious close column, use the second numeric column (first after date)
            close_col = df.columns[0]  # Use first column after Date
        else:
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Could not identify Close price column in {filepath}")
        
        # Extract only the close price and rename to 'price'
        df = df[[close_col]].rename(columns={close_col: 'price'})
        
        # Convert to numeric, handling any non-numeric values
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Calculate logarithmic returns
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Remove rows with NaN values (the first row will have NaN log_return)
        return df.dropna()
        
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
        print("Trying alternative approach...")
        
        try:
            # Based on the image, try a more direct approach
            # Read the CSV file with specific structure shown in the image
            df = pd.read_csv(filepath)
            
            # Print header information for debugging
            print(f"CSV Headers: {df.columns.tolist()}")
            print(f"First row: {df.iloc[0].tolist()}")
            
            # Try to find a structured pattern
            # The image shows "Price" in row 1, columns have values like "Close", "High", etc.
            # Row 2 has "Ticker" with "AAPL" values
            # Row 3 has "Date"
            # Data starts at row 4
            
            # Try to read with skipping the first 3 rows
            df = pd.read_csv(filepath, skiprows=3)
            
            # First column should be Date
            date_col = df.columns[0]
            df = df.rename(columns={date_col: 'Date'})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Second column should be Close price
            if len(df.columns) > 0:
                price_col = df.columns[0]
                df = df[[price_col]].rename(columns={price_col: 'price'})
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                df = df.dropna()
                
                # Calculate log returns
                df['log_return'] = np.log(df['price'] / df['price'].shift(1))
                return df.dropna()
            else:
                raise ValueError(f"Could not extract price data from {filepath}")
                
        except Exception as e2:
            print(f"Alternative approach failed: {str(e2)}")
            raise ValueError(f"Could not process {filepath}. Please check the file format.")

def restructure_stock_data(filepath):
    """
    Enhanced version that keeps all price data columns (Close, High, Low, Open, Volume)
    while addressing the specific format in the image.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    DataFrame with restructured price data indexed by date
    """
    try:
        # First try reading with skiprows=3 based on the image format
        df = pd.read_csv(filepath, skiprows=3)
        
        # Check the columns to see if we got what we expected
        if len(df.columns) < 2:
            # Try alternative approach for different format
            df = pd.read_csv(filepath)
            
            # Check for multi-level headers
            header_rows = []
            for i in range(min(5, len(df))):
                row = df.iloc[i].tolist()
                if any(h in str(row) for h in ['Date', 'Price', 'Close', 'High', 'Low']):
                    header_rows.append(i)
            
            if header_rows:
                # Skip to the row after the last header
                last_header = max(header_rows)
                df = pd.read_csv(filepath, skiprows=last_header+1)
        
        # First column should be Date
        if len(df.columns) > 0:
            date_col = df.columns[0]
            df = df.rename(columns={date_col: 'Date'})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Convert all remaining columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Give appropriate names based on the image if columns appear unnamed
            if any('unnamed' in col.lower() for col in df.columns):
                # Based on the image, we expect columns in this order: Close, High, Low, Open, Volume
                expected_names = ['Close', 'High', 'Low', 'Open', 'Volume']
                
                # Rename columns only if we have the right number
                if len(df.columns) == len(expected_names):
                    df.columns = expected_names
                elif len(df.columns) == 5:  # If we have 5 columns but not the right names
                    df.columns = expected_names
                else:
                    # Partial rename based on position
                    for i, col in enumerate(df.columns):
                        if i < len(expected_names) and 'unnamed' in col.lower():
                            df = df.rename(columns={col: expected_names[i]})
            
            return df.dropna()
        else:
            raise ValueError(f"Could not extract data from {filepath}")
            
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
        
        # One last attempt with a different approach
        try:
            # Try reading the file with header=None to see raw structure
            raw_df = pd.read_csv(filepath, header=None, nrows=10)
            print(f"Raw file structure (first 10 rows):\n{raw_df}")
            
            # Check if the structure matches the image (Price in row 0, Date in row 2)
            if len(raw_df) >= 4:
                if 'Price' in str(raw_df.iloc[0, 0]) or 'PRICE' in str(raw_df.iloc[0, 0]):
                    # Skip first 3 rows (0-based index)
                    df = pd.read_csv(filepath, skiprows=3, header=None)
                    
                    # Assign column names based on the image
                    if len(df.columns) >= 6:
                        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume'] + list(df.columns[6:])
                    elif len(df.columns) >= 1:
                        # Assign what we can
                        columns = ['Date']
                        if len(df.columns) > 1: columns.append('Close')
                        if len(df.columns) > 2: columns.append('High')
                        if len(df.columns) > 3: columns.append('Low')
                        if len(df.columns) > 4: columns.append('Open')
                        if len(df.columns) > 5: columns.append('Volume')
                        df.columns = columns + list(range(len(columns), len(df.columns)))
                    
                    # Set Date as index
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                    
                    # Convert remaining columns to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df.dropna()
            
            raise ValueError(f"Could not determine the structure of {filepath}")
            
        except Exception as e2:
            print(f"All approaches failed: {str(e2)}")
            raise ValueError(f"Could not process {filepath}. Please provide details about the file structure.")

def calculate_log_returns(df):
    """
    Calculate logarithmic returns for price data.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with price data
        
    Returns:
    --------
    DataFrame with added log_return columns
    """
    result_df = df.copy()
    
    # Identify price columns (excluding volume)
    price_cols = [col for col in df.columns if 'volume' not in col.lower()]
    
    # Calculate log returns for each price column
    for col in price_cols:
        return_col = f"{col}_log_return"
        result_df[return_col] = np.log(result_df[col] / result_df[col].shift(1))
    
    return result_df.dropna()
