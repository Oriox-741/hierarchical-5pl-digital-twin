from __future__ import annotations

import unittest

from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT, DiscreteActionMapper


class DiscreteActionMapperContractTest(unittest.TestCase):
    def test_discrete_action_contract_is_48_actions(self) -> None:
        mapper = DiscreteActionMapper()

        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(mapper.action_count, 48)
        mapper.map(47)
        with self.assertRaises(ValueError):
            mapper.map(48)

    def test_all_discrete_actions_round_trip_through_structured_components(self) -> None:
        mapper = DiscreteActionMapper()

        for action_id in range(DISCRETE_ACTION_COUNT):
            with self.subTest(action_id=action_id):
                structured = mapper.map(action_id)
                self.assertEqual(action_id, mapper.encode(structured))


if __name__ == "__main__":
    unittest.main()
