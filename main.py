import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

import numpy as np

# Configuration constants
NUM_INSTRUMENTS = 50
DESIRED_RETURN_TARGET = 3
PIVOT_POINT_WINDOW = 5
MIN_POSITION_CHANGE_THRESHOLD = 5

# Global state variables
current_positions = np.zeros(NUM_INSTRUMENTS, int)
open_positions_per_stock = [[] for _ in range(NUM_INSTRUMENTS)]
# Each position: [position_size, entry_price, lower_bound, upper_bound, position_type]
# position_type: 0 = mean reversion, 1 = trend following
# bound == 0 means bound inactive

def calculate_half_period_statistics(price_array):
    """
    Split price history in half and calculate means/std for each half
    Returns: ([first_half_means, second_half_means], [first_half_stds, second_half_stds])
    """
    num_instruments, num_days = price_array.shape
    
    if num_days <= 1:
        return ([price_array, price_array], [np.zeros(num_instruments), np.zeros(num_instruments)])
    
    # Split data in half
    if num_days % 2 == 0:
        first_half_days = num_days // 2
        second_half_start = first_half_days
    else:
        first_half_days = num_days // 2 + 1
        second_half_start = first_half_days - 1
    
    means = []
    standard_deviations = []
    
    # Calculate statistics for each half
    means.append(price_array[:, :first_half_days].mean(axis=1))
    means.append(price_array[:, second_half_start:].mean(axis=1))
    standard_deviations.append(price_array[:, :first_half_days].std(axis=1, ddof=1))
    standard_deviations.append(price_array[:, second_half_start:].std(axis=1, ddof=1))
    
    return (means, standard_deviations)

def calculate_pivot_points(price_array, days_per_pivot):
    """
    Calculate pivot points for each stock over rolling windows
    Returns: List of pivot points per stock, each pivot: [high, low, pivot_mean]
    pivot_mean = (high + low + closing_price) / 3
    """
    num_instruments, num_days = price_array.shape
    
    if days_per_pivot < 2 or num_instruments == 0 or num_days == 0:
        return [[[price, price, price] for price in stock_prices] for stock_prices in price_array]
    
    all_pivot_points = []
    
    for stock_idx in range(num_instruments):
        stock_pivot_points = []
        current_day = num_days
        
        # Work backwards through time periods
        while current_day > 0:
            period_start = max(0, current_day - days_per_pivot)
            
            if current_day > period_start:
                period_prices = price_array[stock_idx][period_start:current_day]
                high_price = period_prices.max()
                low_price = period_prices.min()
                closing_price = period_prices[-1]
                pivot_mean = (high_price + low_price + closing_price) / 3
                
                stock_pivot_points.append([high_price, low_price, pivot_mean])
            
            current_day = period_start
        
        # Reverse to get chronological order
        stock_pivot_points.reverse()
        all_pivot_points.append(stock_pivot_points)
    
    return all_pivot_points

def calculate_pivot_trend_strength(pivot_points_list, min_consecutive_periods):
    """
    Calculate trend strength from pivot points
    Returns: Price difference between trend endpoints (0 if no consistent trend)
    """
    if len(pivot_points_list) <= 1 or min_consecutive_periods <= 1:
        return 0
    
    current_index = len(pivot_points_list) - 1
    previous_trend_direction = 0
    consecutive_periods_found = 0
    
    while current_index > 0 and consecutive_periods_found < min_consecutive_periods - 1:
        current_pivot_mean = pivot_points_list[current_index][2]
        previous_pivot_mean = pivot_points_list[current_index - 1][2]
        
        # Determine trend direction
        if current_pivot_mean > previous_pivot_mean:
            current_trend_direction = 1  # Uptrend
        elif current_pivot_mean < previous_pivot_mean:
            current_trend_direction = -1  # Downtrend
        else:
            current_trend_direction = 0  # No trend
        
        # Check if trend continues
        if current_trend_direction != 0 and (previous_trend_direction == 0 or current_trend_direction == previous_trend_direction):
            previous_trend_direction = current_trend_direction
            current_index -= 1
            consecutive_periods_found += 1
        else:
            return 0  # Trend broken
    
    # Return price difference if consistent trend found
    if consecutive_periods_found >= min_consecutive_periods - 1:
        return pivot_points_list[-1][2] - pivot_points_list[current_index][2]
    else:
        return 0

def getMyPosition(price_history_so_far):
    """
    Multi-strategy trading algorithm combining mean reversion and trend following
    """
    global current_positions, open_positions_per_stock
    
    num_instruments, num_days = price_history_so_far.shape
    if num_days < 2:
        return np.zeros(num_instruments)
    
    # Define which stocks to actually trade (must match stock_list above!)
    tradeable_stocks = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49]
    
    # Calculate technical indicators
    half_period_stats = calculate_half_period_statistics(price_history_so_far)
    recent_means, recent_std_devs = half_period_stats
    
    all_pivot_points = calculate_pivot_points(price_history_so_far, PIVOT_POINT_WINDOW)
    
    positions = np.zeros(num_instruments)  # Initialize all positions to 0
    
    # Process ONLY the stocks we want to trade
    for stock_idx in tradeable_stocks:  # ← Fixed: only iterate over tradeable stocks
        current_price = price_history_so_far[stock_idx, -1]
        recent_mean = recent_means[1][stock_idx]  # Second half mean
        recent_volatility = recent_std_devs[1][stock_idx]  # Second half std dev
        
        price_deviation = current_price - recent_mean
        abs_price_deviation = abs(price_deviation)
        
        # Calculate trend indicators
        long_term_trend = calculate_pivot_trend_strength(all_pivot_points[stock_idx], 8)
        short_term_trend = calculate_pivot_trend_strength(all_pivot_points[stock_idx], 3)
        
        # Normalize trends to unit direction
        long_term_trend_direction = np.sign(long_term_trend) if long_term_trend != 0 else 0
        short_term_trend_direction = np.sign(short_term_trend) if short_term_trend != 0 else 0
        
        # STRATEGY 1: Mean Reversion (Type 0 positions)
        deviation_threshold_min = 1 * recent_volatility
        deviation_threshold_max = 20 * recent_volatility
        
        if (recent_volatility > 0 and 
            deviation_threshold_min < abs_price_deviation < deviation_threshold_max):
            
            position_size = DESIRED_RETURN_TARGET / recent_volatility * (-price_deviation) / recent_volatility
            
            if price_deviation > 0:  # Price above mean -> short position
                lower_bound = recent_mean
                upper_bound = 0
            else:  # Price below mean -> long position
                lower_bound = 0
                upper_bound = recent_mean
            
            open_positions_per_stock[stock_idx].append([
                position_size, current_price, lower_bound, upper_bound, 0
            ])
        
        # STRATEGY 2: Trend Following (Type 1 positions)
        existing_trend_positions = [pos for pos in open_positions_per_stock[stock_idx] if pos[4] == 1]
        
        if long_term_trend != 0 and len(existing_trend_positions) == 0:
            trend_position_size = 5000 / long_term_trend
            open_positions_per_stock[stock_idx].append([
                trend_position_size, current_price, 0, 0, 1
            ])
        
        # POSITION MANAGEMENT: Review and close positions
        total_position_for_stock = 0
        remaining_positions = []
        
        for position in open_positions_per_stock[stock_idx]:
            position_size, entry_price, lower_bound, upper_bound, position_type = position
            keep_position = False
            
            if position_type == 0:  # Mean reversion position
                within_lower_bound = (current_price > lower_bound) or (lower_bound == 0)
                within_upper_bound = (current_price < upper_bound) or (upper_bound == 0)
                keep_position = within_lower_bound and within_upper_bound
                
                if keep_position:
                    if long_term_trend > 0 and position_size < 0:
                        position[2] += (entry_price - lower_bound) * 0.8
                    elif long_term_trend < 0 and position_size > 0:
                        position[3] -= (upper_bound - entry_price) * 0.8
            
            elif position_type == 1:  # Trend following position
                position_direction = np.sign(position_size)
                keep_position = (position_direction == short_term_trend_direction)
            
            if keep_position:
                remaining_positions.append(position)
                total_position_for_stock += position_size
        
        open_positions_per_stock[stock_idx] = remaining_positions
        
        # Apply transaction cost threshold
        position_change = total_position_for_stock - current_positions[stock_idx]
        
        if abs(position_change) > MIN_POSITION_CHANGE_THRESHOLD:
            positions[stock_idx] = int(total_position_for_stock)
        else:
            positions[stock_idx] = current_positions[stock_idx]
    
    # Update global state (only for tradeable stocks)
    for stock_idx in tradeable_stocks:
        current_positions[stock_idx] = positions[stock_idx]
    
    return positions