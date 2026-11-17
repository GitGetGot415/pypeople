"""Daily food needs shared by simulation decisions and their presentation."""

from constants import SATISFACTION_DAILY_LOSSES, URGENT_FOOD_SATISFACTION


def daily_satisfaction_loss(satisfaction):
    return next((loss for lower, loss in SATISFACTION_DAILY_LOSSES if satisfaction > lower), 0)


def starves_next_day(npc):
    return npc.social <= daily_satisfaction_loss(npc.social)


def urgent_food(npc):
    return npc.social <= URGENT_FOOD_SATISFACTION
