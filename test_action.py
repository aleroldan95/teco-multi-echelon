import numpy as np
import math

num_wh = 4
def get_action_index(wh_origin: int, wh_destination: int):
    """
    Get action index for the given warehouses
    """
    if wh_origin < num_wh and wh_destination < num_wh:
        min_index = min(wh_origin, wh_destination)
        max_index = max(wh_origin, wh_destination)
        return int(num_wh * min_index - min_index * (min_index + 1) / 2) + max_index - min_index - 1
    else:
        raise KeyError('The combination origin_wh - destination_wh entered is not valid')

action = np.array([0] * int(num_wh / 2 * (num_wh - 1)))

stock_by_wh = np.array([7,0,1,3])
forecast_by_wh = np.array([3,2,5,1])

donation_bs_as = max(stock_by_wh[0] - math.ceil(forecast_by_wh[0]), 0)
amount_needed = np.floor(stock_by_wh - forecast_by_wh)

while np.min(amount_needed) < 0 and donation_bs_as != 0:
    max_neg_index = np.where(amount_needed == np.amin(amount_needed))[0][0]
    donation_bs_as -= 1
    amount_needed[max_neg_index] += 1
    stock_by_wh[0] -= donation_bs_as
    stock_by_wh[max_neg_index] += donation_bs_as
    # Add movement to action
    index = get_action_index(wh_origin=0, wh_destination=max_neg_index)
    action[index] += 1

if np.min(amount_needed) < 0:
    while np.amax(amount_needed[1:]) != 0 and np.amin(amount_needed[1:]) < 0:
        amount_needed_without_bs_as = amount_needed[1:]
        max_neg_index = np.where(amount_needed_without_bs_as == np.amin(amount_needed_without_bs_as))[0][0] + 1
        max_pos_index = np.where(amount_needed_without_bs_as == np.amax(amount_needed_without_bs_as))[0][0] + 1
        # amount_donation = min(remaining_stock[max_pos_index], -remaining_stock[max_neg_index])
        amount_donation = 1
        amount_needed[max_pos_index] -= amount_donation
        amount_needed[max_neg_index] += amount_donation
        stock_by_wh[max_pos_index] -= amount_donation
        stock_by_wh[max_neg_index] += amount_donation

        # Add movement to action
        index = get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
        if max_pos_index < max_neg_index:
            action[index] += amount_donation
        else:
            action[index] -= amount_donation

while donation_bs_as != 0 and np.amin(stock_by_wh) == 0:
    amount_needed_without_bs_as = amount_needed[1:]
    max_pos_index = 0
    max_neg_index = np.where(stock_by_wh == 0)[0][0]
    amount_donation = 1
    amount_needed[max_pos_index] -= amount_donation
    stock_by_wh[max_pos_index] -= amount_donation
    stock_by_wh[max_neg_index] += amount_donation
    # Add movement to action
    index = get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
    if max_pos_index < max_neg_index:
        action[index] += amount_donation
    else:
        action[index] -= amount_donation

while np.amax(amount_needed[1:]) > 0 and np.amin(stock_by_wh) == 0:
    amount_needed_without_bs_as = amount_needed[1:]
    max_pos_index = np.where(amount_needed_without_bs_as == np.amax(amount_needed_without_bs_as))[0][0] + 1
    max_neg_index = np.where(stock_by_wh == 0)[0][0]
    amount_donation = 1
    amount_needed[max_pos_index] -= amount_donation
    stock_by_wh[max_pos_index] -= amount_donation
    stock_by_wh[max_neg_index] += amount_donation
    # Add movement to action
    index = get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
    if max_pos_index < max_neg_index:
        action[index] += amount_donation
    else:
        action[index] -= amount_donation

print(action)
print(stock_by_wh)