"""
This is the provided example python code for Chapter twenty nine of the book:
 "Advanced Futures Trading Strategies", by Robert Carver
 https://www.systematicmoney.org/advanced-futures

This code is copyright, Robert Carver 2022.
Shared under https://www.gnu.org/licenses/gpl-3.0.en.html
You may copy, modify, and share this code as long as this header is retained, and you disclose that it has been edited.
This code comes with no warranty, is not guaranteed to be accurate or to match the methodology in the book, and the author is not responsible for any losses that may result from it’s use.

Results may not match the book exactly as different data may be used and the code may contain errors
Results may be different from the corresponding spreadsheet as methods may be slightly different

"""
## Next two lines are optional depending on your IDE
import matplotlib

matplotlib.use("TkAgg")
import datetime
from chapter4 import (
    create_fx_series_given_adjusted_prices_dict,
    calculate_variable_standard_deviation_for_risk_targeting_from_dict,
    calculate_position_series_given_variable_risk_for_dict,
)


from chapter5 import calculate_perc_returns_for_dict_with_costs
from chapter8 import apply_buffering_to_position_dict
from chapter10 import get_data_dict_with_carry
from chapter11 import calculate_position_dict_with_forecast_applied
from chapter28 import transform_into_RV_prices


if __name__ == "__main__":
    ## Get the files from:
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/vix1.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/vix1_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/vix2.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/vix2_carry.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/vix3.csv
    # https://gitfront.io/r/user-4000052/iTvUZwEUN2Ta/AFTS-CODE/blob/vix3_carry.csv
    (
        adjusted_prices_dict,
        current_prices_dict,
        carry_prices_dict,
    ) = get_data_dict_with_carry(instrument_list=["vix1", "vix2", "vix3"])

    multipliers = dict(vix1=1000, vix2=1000, edollar3=1000)
    risk_target_tau = 0.1
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    ###### TRIPLETS
    instrument_code_list = ["vix2", "vix1", "vix3"]
    ratio_list = [-1, 0.5, 0.5]
    rv_instrument_name = "VIX123"

    new_data = transform_into_RV_prices(
        adjusted_prices_dict=adjusted_prices_dict,
        carry_prices_dict=carry_prices_dict,
        multipliers=multipliers,
        fx_series_dict=fx_series_dict,
        current_prices_dict=current_prices_dict,
        instrument_code_list=instrument_code_list,
        ratio_list=ratio_list,
        rv_instrument_name=rv_instrument_name,
        start_date=datetime.datetime(2008, 1, 1),
    )

    capital = 2000000

    idm = 1.0
    instrument_weights = {rv_instrument_name: 1.0}
    cost_per_contract_dict = {rv_instrument_name: 30}

    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=new_data["adjusted_prices_dict"],
        current_prices=new_data["current_prices_dict"],
        use_perc_returns=False,
    )

    average_position_contracts_dict = (
        calculate_position_series_given_variable_risk_for_dict(
            capital=capital,
            risk_target_tau=risk_target_tau,
            idm=idm,
            weights=instrument_weights,
            std_dev_dict=std_dev_dict,
            fx_series_dict=new_data["fx_series_dict"],
            multipliers=new_data["multipliers"],
        )
    )

    ## Assumes equal forecast weights and we use all rules for both instruments
    rules_spec = [
        dict(function="carry", span=5),
        dict(function="carry", span=20),
        dict(function="carry", span=60),
        dict(function="carry", span=120),
        dict(function="ewmac", fast_span=16),
        dict(function="ewmac", fast_span=32),
        dict(function="ewmac", fast_span=64),
    ]
    position_contracts_dict = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=new_data["adjusted_prices_dict"],
        carry_prices_dict=new_data["carry_prices_dict"],
        std_dev_dict=std_dev_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        rule_spec=rules_spec,
    )

    buffered_position_dict = apply_buffering_to_position_dict(
        position_contracts_dict=position_contracts_dict,
        average_position_contracts_dict=average_position_contracts_dict,
    )

    perc_return_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict,
        fx_series=new_data["fx_series_dict"],
        multipliers=new_data["multipliers"],
        capital=capital,
        adjusted_prices=new_data["adjusted_prices_dict"],
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    ###### SPREADS
    instrument_code_list = ["vix2", "vix1"]
    ratio_list = [-1, 1]
    rv_instrument_name = "VIX12"

    new_data = transform_into_RV_prices(
        adjusted_prices_dict=adjusted_prices_dict,
        carry_prices_dict=carry_prices_dict,
        multipliers=multipliers,
        fx_series_dict=fx_series_dict,
        current_prices_dict=current_prices_dict,
        instrument_code_list=instrument_code_list,
        ratio_list=ratio_list,
        rv_instrument_name=rv_instrument_name,
    )

    capital = 2000000

    idm = 1.0
    instrument_weights = {rv_instrument_name: 1.0}
    cost_per_contract_dict = {rv_instrument_name: 30}

    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=new_data["adjusted_prices_dict"],
        current_prices=new_data["current_prices_dict"],
    )

    average_position_contracts_dict = (
        calculate_position_series_given_variable_risk_for_dict(
            capital=capital,
            risk_target_tau=risk_target_tau,
            idm=idm,
            weights=instrument_weights,
            std_dev_dict=std_dev_dict,
            fx_series_dict=new_data["fx_series_dict"],
            multipliers=new_data["multipliers"],
        )
    )

    ## Assumes equal forecast weights and we use all rules for both instruments
    rules_spec = [
        dict(function="carry", span=5),
        dict(function="carry", span=20),
        dict(function="carry", span=60),
        dict(function="carry", span=120),
        dict(function="ewmac", fast_span=16),
        dict(function="ewmac", fast_span=32),
        dict(function="ewmac", fast_span=64),
    ]
    position_contracts_dict = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=new_data["adjusted_prices_dict"],
        carry_prices_dict=new_data["carry_prices_dict"],
        std_dev_dict=std_dev_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        rule_spec=rules_spec,
    )

    buffered_position_dict = apply_buffering_to_position_dict(
        position_contracts_dict=position_contracts_dict,
        average_position_contracts_dict=average_position_contracts_dict,
    )

    perc_return_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict,
        fx_series=new_data["fx_series_dict"],
        multipliers=new_data["multipliers"],
        capital=capital,
        adjusted_prices=new_data["adjusted_prices_dict"],
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )
