"""
Time-Aware SmartController
The Expert Controller for Behavior Cloning
"""

class TimeAwareSC:
    """
    Adaptive heating margins by temperature zone + Time-aware finishing
    Total Cost on Benchmark: 159.47 (8.7% improvement over original SC)
    """
    
    def decide(self, temp, units):
        active = [u for u in units.values() if u['state'] != 'FINISHED']
        if not active: return 0.0
        
        heating = [u for u in active if u['state'] == 'HEATING']
        holding = [u for u in active if u['state'] == 'HOLDING']
        
        # Get max target and select margins
        all_targets = [u['target'] for u in active]
        max_target = max(all_targets)
        
        if max_target < 100:
            heat_margin, hold_margin = 8, 1
        elif max_target < 150:
            heat_margin, hold_margin = 20, 2
        else:
            heat_margin, hold_margin = 40, 2
        
        # Phase 1: HEATING (Bang-Bang)
        if heating:
            heat_target = max([u['target'] for u in heating])
            target_boiler = heat_target + heat_margin
            
            if temp < target_boiler:
                return 100.0  # Full power
            else:
                return 0.0    # Coast
        
        # Phase 2: HOLDING (Time-Aware)
        if holding:
            hold_target = max([u['target'] for u in holding])
            min_remaining = min([u['duration_left'] for u in holding])
            
            # TIME-AWARE FINISHING LOGIC
            if min_remaining <= 15:
                # Last 15 seconds: stop heating, coast to finish
                return 0.0
            elif min_remaining <= 30:
                # Last 30 seconds: minimal power
                target_boiler = hold_target  # Exact target, no margin
                gap = target_boiler - temp
                if gap > 1: return 15.0
                else: return 0.0
            else:
                # Normal holding
                target_boiler = hold_target + hold_margin
                gap = target_boiler - temp
                
                boiler_loss = (temp - 25.0) * 4.0
                required_kw = boiler_loss / (50.0 * 0.9)
                required_pct = (required_kw / 100.0) * 100.0 * 1.2
                
                if gap > 3:
                    return min(100, max(40, required_pct))
                elif gap > 0:
                    return min(100, max(15, required_pct))
                elif gap < -2:
                    return 0.0
                else:
                    return min(100, max(10, required_pct))
        
        return 0.0
