import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class TradingEnv(gym.Env):
    """
    A 3-stock (MSFT, IBM, QCOM) trading environment.

    State: [# of stock owned, current stock prices, cash in hand]
      - array of length n_stock * 2 + 1
      - price is discretized (to integer) to reduce state space
      - use close price for each stock
      - cash in hand is evaluated at each step based on action performed

      - when selling, sell all the shares
      - when buying, buy as many as cash in hand allows
      - if buying multiple stock, equally distribute cash in hand and then utilize the balance
    """

    def __init__(self, train_data, init_invest=20000):
        # data
        # round up to integer to reduce state space
        self.stock_price_history = np.around(train_data)
        print(self.stock_price_history.shape)
        self.n_stock, self.n_step = self.stock_price_history.shape

        # instance attributes
        self.init_invest = init_invest
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # action space
        self.action_space = spaces.Discrete(9**self.n_stock)
        print("action_space = ", self.action_space)

        # observation space: give estimates in order to sample and build scaler
        stock_max_price = self.stock_price_history.max(axis=1)
        print("stock_max_price = ", stock_max_price)
        stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
        print("stock_range = ", stock_range)
        price_range = [[0, mx] for mx in stock_max_price]
        print("price_range = ", price_range)
        cash_in_hand_range = [[0, init_invest * 2]]
        print("cash_in_hand_range = ", cash_in_hand_range)
        self.observation_space = spaces.MultiDiscrete(
            stock_range + price_range + cash_in_hand_range)
        print("observation_space = ", self.observation_space)

        # seed and start
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.cur_step = 0
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.cash_in_hand = self.init_invest
        return self._get_obs()

    def _step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1
        # update price
        self.stock_price = self.stock_price_history[:, self.cur_step]
        # print("self.stock_price = ", self.stock_price)
        self._trade(action)
        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = []
        obs.extend(self.stock_owned)
        obs.extend(list(self.stock_price))
        obs.append(self.cash_in_hand)
        return obs

    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        # Action: hold (0)
        # (1),buy 5% (2) ,buy 10% (3), buy 15% (4),buy 25%
        # (5),sell 25% (6) ,sell 50% (7), sell 75% (8),sell 100%

        # all combo to sell(0), hold(1), or buy(2) stocks

        action_combo = list(map(list, itertools.product(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], repeat=self.n_stock)))
        action_vec = action_combo[action]
        # print("action_vec = ", action_vec)
        # one pass to get sell/buy index

        for i, a in enumerate(action_vec):
            if a >= 1 and a <= 4:
                need_cash_in_market = (self.init_invest * 0.05 * a)
                if self.cash_in_hand > need_cash_in_market and need_cash_in_market > self.stock_price[i]:
                    self.stock_owned[i] += (need_cash_in_market //
                                            self.stock_price[i])
                    self.cash_in_hand -= need_cash_in_market
            elif a > 4:
                if self.stock_owned[i] > 0:
                    self.cash_in_hand += self.stock_price[i] * \
                        self.stock_owned[i] * 0.25 * (a-4)
                    self.stock_owned[i] *= (1 - (0.25*(a-4)))

        # print("buy_index = ", buy_index)
        # print("self.stock_owned = ", self.stock_owned)
        # print("self.cash_in_hand = ", self.cash_in_hand)
