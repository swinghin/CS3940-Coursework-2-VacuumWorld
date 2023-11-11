#!/usr/bin/env python3
from typing import Iterable
from pyoptional.pyoptional import PyOptional

from vacuumworld import run
from vacuumworld.model.actions.vwactions import VWAction
from vacuumworld.model.actions.vwidle_action import VWIdleAction
from vacuumworld.model.actions.vwmove_action import VWMoveAction
from vacuumworld.model.actions.vwturn_action import VWTurnAction
from vacuumworld.model.environment.vwlocation import VWLocation
from vacuumworld.common.vwdirection import VWDirection
from vacuumworld.model.actions.vweffort import VWActionEffort
from vacuumworld.model.actor.mind.surrogate.vwactor_mind_surrogate import (
    VWActorMindSurrogate,
)
import numpy as np


class ZigZagMind(VWActorMindSurrogate):
    def __init__(self) -> None:
        super(ZigZagMind, self).__init__()

        # Variable to store current stage:
        #  -1: just dropped onto grid
        #   0: go to bottom right corner,
        #   1: zigzag leftwards and upwards until top left corner
        #   2: idle after complete exploration
        self.__stage: int = -1

        # True if agent is found to be at east edge
        self.__start_at_east_edge: bool = False
        self.__start_at_south_edge: bool = False

        # Agent self map: n x n array
        #  -1: unexplored cell
        #   0: empty cell
        #   1: orange dirt cell
        #   2: green dirt cell
        self.__map: list[list[int]] = []

        # Grid size
        self.__n: int = -1

        # Check booleans if at correct spot to start scan
        self.__at_one_cell_from_east_edge: bool = False
        self.__at_one_cell_from_south_edge: bool = False
        self.__started_scan: bool = False

        # Zigzag scan control variable
        #   -1: not started
        # even: scan west
        #  odd: scan east
        self.__scan_pass: int = -1
        # Intermission scan (scan north between west/east scans)
        self.__scan_inter: bool = False

    ### REVISE FUNCTIONS ###

    def __is_one_step_from_east_wall(self) -> bool:
        return (
            self.get_own_appearance().is_facing_east()
            and self.get_latest_observation().is_wall_one_step_ahead()
        )

    def __is_one_step_from_south_wall(self) -> bool:
        return (
            self.get_own_appearance().is_facing_south()
            and self.get_latest_observation().is_wall_one_step_ahead()
        )

    def __is_one_step_from_west_wall(self) -> bool:
        return (
            self.get_own_appearance().is_facing_west()
            and self.get_latest_observation().is_wall_one_step_ahead()
        )

    def __is_one_step_from_north_wall(self) -> bool:
        return (
            self.get_own_appearance().is_facing_north()
            and self.get_latest_observation().is_wall_one_step_ahead()
        )

    def __revise_stage_n1(self) -> None:
        # if just dropped on grid
        # first, check if agent is at east edge
        if (
            self.get_latest_observation()
            .get_center()
            .or_else_raise()
            .has_wall_on_east()
        ):
            # if at east edge, size n is found
            self.__n = self.get_own_position().get_x() + 1
            self.__start_at_east_edge = True

        # second, check if agent is at south edge
        if (
            self.get_latest_observation()
            .get_center()
            .or_else_raise()
            .has_wall_on_south()
        ):
            # if at south edge, size n is found
            self.__n = self.get_own_position().get_y() + 1
            self.__start_at_south_edge = True

        # move to stage 0
        self.__stage = 0

    def __revise_stage_0(self) -> None:
        # if not start at east edge, should be going east until one cell away from east edge
        if not self.__start_at_east_edge:
            if self.__is_one_step_from_east_wall():
                self.__n = self.get_own_position().get_x() + 2
                self.__at_one_cell_from_east_edge = True
        else:
            if self.get_own_position().get_x() == (self.__n - 2):
                self.__at_one_cell_from_east_edge = True

        # if not start at south edge, should be going north until one cell away from south edge
        if not self.__start_at_south_edge:
            if self.__is_one_step_from_south_wall():
                self.__at_one_cell_from_south_edge = True
        else:
            if self.get_own_position().get_y() == (self.__n - 2):
                self.__at_one_cell_from_south_edge = True

        # if current at bottom right, go to stage 1
        if self.__at_one_cell_from_east_edge and self.__at_one_cell_from_south_edge:
            self.__stage = 1
            print(f"Grid size n={self.__n}")

    def __is_map_populated(self) -> bool:
        # loop through agent self map, return if all cells explored (no -1 values in 2d array)
        for x in range(self.__n):
            for y in range(self.__n):
                if self.__map and self.__map[x][y] == -1:
                    return False
        return True

    def __observe_cell(self, cell: VWLocation) -> None:
        # get x y coord of cell
        cell_x: int = cell.get_coord().get_x()
        cell_y: int = cell.get_coord().get_y()

        # check if cell has dirt, if so, find its colour and save to map
        if cell.has_dirt():
            dirt_colour: str = str(
                cell.get_dirt_appearance().or_else_raise().get_colour()
            )
            if dirt_colour == "orange":
                self.__map[cell_x][cell_y] = 1
            elif dirt_colour == "green":
                self.__map[cell_x][cell_y] = 2
        # if no dirt, save to map as empty
        else:
            self.__map[cell_x][cell_y] = 0

    def __scan_grid(self) -> None:
        # if scanning west and one step from west wall, start going north for at most 3 cells
        if self.__scan_pass % 2 == 0 and self.__is_one_step_from_west_wall():
            self.__scan_inter = True
            self.__scan_inter_start = self.get_own_position().get_y()

        # similar to west wall condition above, start going north for at most 3 cells
        if self.__scan_pass % 2 == 1 and self.__is_one_step_from_east_wall():
            self.__scan_inter = True
            self.__scan_inter_start = self.get_own_position().get_y()

        # if need to go north, stop if one step away from north wall or 3 cells above last pass
        if self.__scan_inter and (
            self.__scan_inter_start - self.get_own_position().get_y() > 2
            or self.__is_one_step_from_north_wall()
        ):
            self.__scan_inter = False
            self.__scan_pass += 1

        # each pass, get a list of 6 cells ahead of agent
        observed_cells: list[
            PyOptional[VWLocation]
        ] = self.get_latest_observation().get_locations_in_order()
        # observe the 6 cells
        for cell in observed_cells:
            cell = cell.or_else_raise()
            self.__observe_cell(cell)

    def __revise_stage_1(self) -> None:
        # initialise agent self map with size n x n
        if not self.__map:
            self.__map = [[-1 for _ in range(self.__n)] for _ in range(self.__n)]

        # start scan if oriented east at bottom right
        if not self.__started_scan and self.get_own_appearance().is_facing_east():
            self.__started_scan = True
            self.__scan_pass = 0

        if self.__started_scan:
            self.__scan_grid()

        # move to stage 2 idle if map populated
        if self.__is_map_populated():
            self.__stage = 2

    def __revise_stage_2(self) -> None:
        # after exploration is done print out grid size and agent's internal map
        print(f"Grid size n = {self.__n}")

        # flip horizontal and rotate -90deg to flip x y axis in array representation
        print("Agent internal map: 0 = empty cell, 1 = orange dirt, 2 = green dirt")
        print(np.rot90(np.flip(np.array(self.__map), axis=1), 1))

    def revise(self) -> None:
        if self.__stage == 1:
            self.__revise_stage_1()
        elif self.__stage == 0:
            self.__revise_stage_0()
        elif self.__stage == -1:
            self.__revise_stage_n1()
        elif self.__stage == 2:
            self.__revise_stage_2()

    ### DECIDE FUNCTIONS ###

    def __go_towards_east(self) -> Iterable[VWAction]:
        if self.get_own_appearance().is_facing_east():
            return [VWMoveAction()]
        elif self.get_own_appearance().is_facing_north():
            return [VWTurnAction(VWDirection.right)]
        else:
            return [VWTurnAction(VWDirection.left)]

    def __go_towards_south(self) -> Iterable[VWAction]:
        if self.get_own_appearance().is_facing_south():
            return [VWMoveAction()]
        elif self.get_own_appearance().is_facing_east():
            return [VWTurnAction(VWDirection.right)]
        else:
            return [VWTurnAction(VWDirection.left)]

    def __go_towards_west(self) -> Iterable[VWAction]:
        if self.get_own_appearance().is_facing_west():
            return [VWMoveAction()]
        elif self.get_own_appearance().is_facing_south():
            return [VWTurnAction(VWDirection.right)]
        else:
            return [VWTurnAction(VWDirection.left)]

    def __go_towards_north(self) -> Iterable[VWAction]:
        if self.get_own_appearance().is_facing_north():
            return [VWMoveAction()]
        elif self.get_own_appearance().is_facing_west():
            return [VWTurnAction(VWDirection.right)]
        else:
            return [VWTurnAction(VWDirection.left)]

    def __explore(self) -> Iterable[VWAction]:
        # after scan started, go left and start scanning
        # if scan_pass is even go west, odd go east, if intermission scan go north
        if self.__started_scan:
            if self.__scan_inter:
                return self.__go_towards_north()
            if self.__scan_pass % 2 == 0:
                return self.__go_towards_west()
            else:
                return self.__go_towards_east()

        # if start scan criteria not met, try to turn east
        if not self.get_own_appearance().is_facing_east():
            return self.__go_towards_east()

        return [VWIdleAction()]

    def __go_bottom_right(self) -> Iterable[VWAction]:
        # At this line robot should be one cell away from east edge
        # now go until one cell away from south edge
        if not self.__at_one_cell_from_south_edge:
            if not self.__start_at_south_edge:
                return self.__go_towards_south()
            else:
                return self.__go_towards_north()

        # If this code line is reachable, robot is oriented properly
        # now go until one cell away from east edge
        if not self.__at_one_cell_from_east_edge:
            if not self.__start_at_east_edge:
                return self.__go_towards_east()
            else:
                return self.__go_towards_west()

        return [VWIdleAction()]

    def decide(self) -> Iterable[VWAction]:
        if self.__stage == 1:
            return self.__explore()
        elif self.__stage == 0:
            return self.__go_bottom_right()

        return [VWIdleAction()]


if __name__ == "__main__":
    run(
        default_mind=ZigZagMind(),
        efforts=VWActionEffort.REASONABLE_EFFORTS,
        skip=True,
        speed=0.8,
    )
