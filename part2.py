#!/usr/bin/env python3
import math
import json
from typing import Iterable
from pyoptional.pyoptional import PyOptional

from pystarworldsturbo.common.message import BccMessage
from vacuumworld import run
from vacuumworld.common.vwcoordinates import VWCoord
from vacuumworld.common.vwobservation import VWObservation
from vacuumworld.common.vworientation import VWOrientation
from vacuumworld.common.vwdirection import VWDirection
from vacuumworld.model.actions.vwactions import VWAction
from vacuumworld.model.actions.vwclean_action import VWCleanAction
from vacuumworld.model.actions.vwidle_action import VWIdleAction
from vacuumworld.model.actions.vwmove_action import VWMoveAction
from vacuumworld.model.actions.vwspeak_action import VWSpeakAction
from vacuumworld.model.actions.vwturn_action import VWTurnAction
from vacuumworld.model.actions.vwbroadcast_action import VWBroadcastAction
from vacuumworld.model.actor.appearance.vwactor_appearance import VWActorAppearance
from vacuumworld.model.environment.vwlocation import VWLocation
from vacuumworld.model.actions.vweffort import VWActionEffort
from vacuumworld.model.actor.mind.surrogate.vwactor_mind_surrogate import (
    VWActorMindSurrogate,
)


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
        self.__dirt_loc: dict[str, list[str]] = {"orange": [], "green": []}

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

        # Announcement variables
        self.__announcement: str = ""
        self.__this_message: tuple[str, str] = ("", "")
        self.__messages: dict[str, str] = {}
        self.__agent_list: list[dict[str, str]] = []
        self.__announced_dirt_loc: bool = False

        self.__coord_to_go: VWCoord = VWCoord(-1, -1)
        self.__direction_to_go: VWOrientation = VWOrientation.north
        self.__should_clean: bool = False
        self.__now_cleaning_colour: str = ""

    ### REVISE FUNCTIONS ###

    def __is_one_step_from_wall(self, orientation: VWOrientation) -> bool:
        return (
            self.get_own_appearance().is_facing(orientation)
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
            if self.__is_one_step_from_wall(VWOrientation.east):
                self.__n = self.get_own_position().get_x() + 2
                self.__at_one_cell_from_east_edge = True
        else:
            if self.get_own_position().get_x() == (self.__n - 2):
                self.__at_one_cell_from_east_edge = True

        # if not start at south edge, should be going north until one cell away from south edge
        if not self.__start_at_south_edge:
            if self.__is_one_step_from_wall(VWOrientation.south):
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
        if self.__scan_pass % 2 == 0 and self.__is_one_step_from_wall(
            VWOrientation.west
        ):
            self.__scan_inter = True
            self.__scan_inter_start = self.get_own_position().get_y()

        # similar to west wall condition above, start going north for at most 3 cells
        if self.__scan_pass % 2 == 1 and self.__is_one_step_from_wall(
            VWOrientation.east
        ):
            self.__scan_inter = True
            self.__scan_inter_start = self.get_own_position().get_y()

        # if need to go north, stop if one step away from north wall or 3 cells above last pass
        if self.__scan_inter and (
            self.__scan_inter_start - self.get_own_position().get_y() > 2
            or self.__is_one_step_from_wall(VWOrientation.north)
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
        # print(f"Grid size n = {self.__n}")

        # flip horizontal and rotate -90deg to flip x y axis in array representation
        # print("Agent internal map: 0 = empty cell, 1 = orange dirt, 2 = green dirt")
        # grid_str: str = ""
        # for x in range(self.__n):
        #     for y in range(self.__n):
        #         grid_str += f"{self.__map[y][x]} "
        #     grid_str += "\n"
        # print(grid_str)

        # build arrays of coloured dirt
        if not self.__announced_dirt_loc:
            self.__prepare_dirt_dict()

    def __prepare_dirt_dict(self) -> None:
        self.__dirt_loc = {"orange": [], "green": []}
        # loop through agent self map, append
        for x in range(self.__n):
            for y in range(self.__n):
                if self.__map[x][y] == 1:
                    self.__dirt_loc["orange"].append(f"{x},{y}")
                if self.__map[x][y] == 2:
                    self.__dirt_loc["green"].append(f"{x},{y}")

        announcement: dict[str, list[str]] = {
            "command": ["clean"],
            "orange": self.__dirt_loc["orange"],
            "green": self.__dirt_loc["green"],
        }
        self.__announcement = json.dumps(announcement)
        self.__announced_dirt_loc = True

    def __clear_announcement(self) -> None:
        self.__announcement = ""

    def __listen_roll_call(self) -> None:
        for message in self.get_latest_received_messages():
            message_content: dict[str, str] = json.loads(str(message.get_content()))
            if message_content not in self.__agent_list:
                self.__agent_list.append(message_content)

    def __prepare_roll_call(self) -> None:
        announcement: dict[str, list[str]] = {
            "command": ["rollcall"],
        }
        self.__announcement = json.dumps(announcement)

    def __add_message(self, actor_id: str, message: str) -> None:
        self.__messages[actor_id] = message

    def __prepare_message(self) -> None:
        if self.__messages:
            self.__this_message = list(self.__messages.items())[0]
            agent_id = self.__this_message[0]
            del self.__messages[agent_id]
        else:
            self.__this_message = ("", "")

    def __find_cell_for_agent(
        self,
        my_orient: VWOrientation,
        actor_orient: VWOrientation,
        left_loc: PyOptional[VWLocation],
        right_loc: PyOptional[VWLocation],
    ) -> VWCoord:
        own_pos: VWCoord = self.get_own_position()
        print(my_orient)
        if my_orient == VWOrientation.east:
            one_step_forward_coord = VWCoord(own_pos.get_x() + 2, (own_pos.get_y()))
        elif my_orient == VWOrientation.south:
            one_step_forward_coord = VWCoord(own_pos.get_x(), (own_pos.get_y() + 2))
        elif my_orient == VWOrientation.west:
            one_step_forward_coord = VWCoord(own_pos.get_x() - 2, (own_pos.get_y()))
        else:
            one_step_forward_coord = VWCoord(own_pos.get_x(), (own_pos.get_y() - 2))

        # if self and other agent face same direction, find empty fl or fr cell
        if (
            actor_orient == my_orient
            or actor_orient == my_orient.get_opposite()
            or actor_orient == my_orient.get_left()
        ):
            if self.__check_valid_empty_cell(left_loc):
                return left_loc.or_else_raise().get_coord()
            elif self.__check_valid_empty_cell(right_loc):
                return right_loc.or_else_raise().get_coord()
            else:
                return one_step_forward_coord
        elif actor_orient == my_orient.get_right():
            if self.__check_valid_empty_cell(right_loc):
                return right_loc.or_else_raise().get_coord()
            elif self.__check_valid_empty_cell(left_loc):
                return left_loc.or_else_raise().get_coord()
            else:
                return one_step_forward_coord
        return VWCoord(-1, -1)

    def __ask_agent_to_go(
        self, actor: VWActorAppearance, observation: VWObservation
    ) -> None:
        # find a spot for actor to go
        goto: VWCoord = self.__find_cell_for_agent(
            self.get_own_orientation(),
            actor.get_orientation(),
            observation.get_forwardleft(),
            observation.get_forwardright(),
        )
        
        if goto != VWCoord(-1, -1):
            instruction: dict[str, str | list[str]] = {
                "command": ["getout"],
                "goto": [f"{goto.get_x()},{goto.get_y()}"],
            }
            self.__add_message(actor.get_id(), json.dumps(instruction))

    def __check_agent_cell(self, location: PyOptional[VWLocation]) -> bool:
        return not location.is_empty() and location.or_else_raise().has_actor()

    def __check_valid_empty_cell(self, location: PyOptional[VWLocation]) -> bool:
        return not location.is_empty() and not location.or_else_raise().has_actor()

    def __detect_obstacle(self) -> None:
        observation: VWObservation = self.get_latest_observation()
        forward_location: PyOptional[VWLocation] = observation.get_forward()
        if self.__check_agent_cell(forward_location):
            actor: VWActorAppearance = (
                forward_location.or_else_raise().get_actor_appearance().or_else_raise()
            )
            self.__ask_agent_to_go(actor, observation)

    # print dirt loc
    def __listen_dirt_update(self) -> None:
        for message in self.get_latest_received_messages():
            message_content: dict[str, str] = json.loads(str(message.get_content()))
            colour: str = message_content["colour"]
            coord = message_content["coord"]
            if coord in self.__dirt_loc[colour]:
                self.__dirt_loc[colour].remove(coord)

        # print(self.__dirt_loc)
        if not self.__dirt_loc["orange"] and not self.__dirt_loc["green"]:
            self.__stage = 3

    def __calc_direction_to_go(self) -> VWOrientation:
        now_coord: VWCoord = self.get_own_position()
        now_orientation: VWOrientation = self.get_own_orientation()
        delta_x: int = self.__coord_to_go.get_x() - now_coord.get_x()
        delta_y: int = self.__coord_to_go.get_y() - now_coord.get_y()

        # if delta_x=0,g o N/S based on delta_y
        if delta_x == 0:
            return VWOrientation.north if delta_y < 0 else VWOrientation.south
        # if delta_y=0, go E/W based on delta_x
        elif delta_y == 0:
            return VWOrientation.west if delta_x < 0 else VWOrientation.east

        if (
            now_orientation == VWOrientation.north
            or now_orientation == VWOrientation.south
        ):
            return VWOrientation.north if delta_y < 0 else VWOrientation.south
        else:
            return VWOrientation.west if delta_x < 0 else VWOrientation.east

    def __get_coord_distance(
        self,
        agent_coord: VWCoord,
        target_coord: VWCoord,
    ) -> float:
        # use pythagorean thm to get distance between agent and target cell
        delta_x: int = agent_coord.get_x() - target_coord.get_x()
        delta_y: int = agent_coord.get_y() - target_coord.get_y()
        distance = math.sqrt(delta_x**2 + delta_y**2)
        return distance

    def __calc_colour_to_clean(self) -> None:
        self.__now_cleaning_colour = (
            "orange"
            if len(self.__dirt_loc["orange"]) > len(self.__dirt_loc["green"])
            else "green"
        )

    def __get_nearest_coord(self) -> VWCoord:
        nearest_coord: VWCoord = VWCoord(-1, -1)
        nearest_distance: float = math.inf

        agent_coord = self.get_own_position()
        self.__calc_colour_to_clean()

        for dirt_coord in self.__dirt_loc[self.__now_cleaning_colour]:
            x, y = dirt_coord.split(",")
            dirt_vwcoord: VWCoord = VWCoord(int(x), int(y))
            dirt_distance: float = self.__get_coord_distance(agent_coord, dirt_vwcoord)
            if dirt_distance < nearest_distance:
                nearest_distance = dirt_distance
                nearest_coord = dirt_vwcoord

        return nearest_coord

    def __find_coord_to_go(self) -> None:
        self.__coord_to_go = self.__get_nearest_coord()

    def __prepare_help(self) -> None:
        # if not yet arrived at target coordinate
        if self.get_own_position() != self.__coord_to_go:
            # if target coordinate is invalid, find somewhere to go
            if self.__coord_to_go == VWCoord(-1, -1):
                self.__find_coord_to_go()
            # then, find which direction to go
            self.__direction_to_go = self.__calc_direction_to_go()
        # if arrived at target coordinate
        else:
            # first check if target coordinate has dirt, if so, it needs to be cleaned
            if self.get_latest_observation().get_center().or_else_raise().has_dirt():
                self.__should_clean = True
            # if no dirt, remove current coordinate from cleaning list,
            # tell master this place is cleaned, then
            # find another place to go
            else:
                self_coord = f"{self.get_own_position().get_x()},{self.get_own_position().get_y()}"
                if self_coord in self.__dirt_loc[self.__now_cleaning_colour]:
                    self.__dirt_loc[self.__now_cleaning_colour].remove(self_coord)
                self.__should_clean = False
                self.__find_coord_to_go()

    def __ask_agent_to_ignore(self) -> None:
        instruction: dict[str, str | list[str]] = {
            "command": ["ignore"],
            "coord": [f"{self.__coord_to_go.get_x()},{self.__coord_to_go.get_y()}"],
        }
        for agent in self.__agent_list:
            if agent["colour"] == self.__now_cleaning_colour:
                self.__add_message(agent["id"], json.dumps(instruction))

    def revise(self) -> None:
        self.__clear_announcement()

        if not self.__agent_list:
            self.__prepare_roll_call()
            self.__listen_roll_call()
        else:
            self.__detect_obstacle()
            self.__prepare_message()

        if self.__stage == 1:
            self.__revise_stage_1()
        elif self.__stage == 0:
            self.__revise_stage_0()
        elif self.__stage == -1:
            self.__revise_stage_n1()
        elif self.__stage == 2:
            self.__revise_stage_2()
            self.__listen_dirt_update()
            self.__prepare_help()
            self.__ask_agent_to_ignore()
            print(
                f"white at {self.get_own_position()} going {self.__coord_to_go}, now cleaning {self.__now_cleaning_colour}, should clean={self.__should_clean}"
            )

    ### DECIDE FUNCTIONS ###

    def __shout(self) -> VWAction:
        return VWBroadcastAction(
            message=self.__announcement, sender_id=self.get_own_id()
        )

    def __whisper(self, recipient_id: str, message: str) -> VWAction:
        return VWSpeakAction(
            message=message,
            recipients=[recipient_id],
            sender_id=self.get_own_id(),
        )

    def __go_and_speak(self, orientation: VWOrientation) -> Iterable[VWAction]:
        action: list[VWAction] = [self.__go_towards(orientation)]

        # check if anything to broadcast
        if self.__announcement:
            action.append(self.__shout())
        else:
            # check if anything to speak to agents
            if self.__this_message:
                agent_id, message = self.__this_message
                action.append(self.__whisper(agent_id, message))

        return action

    def __go_towards(self, orientation: VWOrientation) -> VWAction:
        if self.get_own_appearance().is_facing(orientation):
            return VWMoveAction()
        elif self.get_own_appearance().is_facing(orientation.get_left()):
            return VWTurnAction(VWDirection.right)
        else:
            return VWTurnAction(VWDirection.left)

    def __goto_coord(self, coord: VWCoord) -> VWAction:
        if self.get_own_position() != coord:
            return self.__go_towards(self.__direction_to_go)

        # if own position is already at coord, do nothing
        return VWIdleAction()

    def __clean(self) -> VWAction:
        return VWCleanAction()

    def __explore(self) -> Iterable[VWAction]:
        # after scan started, go left and start scanning
        # if scan_pass is even go west, odd go east, if intermission scan go north
        if self.__started_scan:
            if self.__scan_inter:
                return self.__go_and_speak(VWOrientation.north)
            if self.__scan_pass % 2 == 0:
                return self.__go_and_speak(VWOrientation.west)
            else:
                return self.__go_and_speak(VWOrientation.east)

        # if start scan criteria not met, try to turn east
        if not self.get_own_appearance().is_facing_east():
            return self.__go_and_speak(VWOrientation.east)

        return [VWIdleAction()]

    def __go_bottom_right(self) -> Iterable[VWAction]:
        # At this line robot should be one cell away from east edge
        # now go until one cell away from south edge
        if not self.__at_one_cell_from_south_edge:
            if not self.__start_at_south_edge:
                return self.__go_and_speak(VWOrientation.south)
            else:
                return self.__go_and_speak(VWOrientation.north)

        # If this code line is reachable, robot is oriented properly
        # now go until one cell away from east edge
        if not self.__at_one_cell_from_east_edge:
            if not self.__start_at_east_edge:
                return self.__go_and_speak(VWOrientation.east)
            else:
                return self.__go_and_speak(VWOrientation.west)

        return [VWIdleAction()]

    def __help_clean(self) -> Iterable[VWAction]:
        action: list[VWAction] = []

        if self.get_own_position() == self.__coord_to_go and self.__should_clean:
            action.append(self.__clean())

        elif self.__coord_to_go != VWCoord(-1, -1):
            action.append(self.__goto_coord(self.__coord_to_go))

        # check if anything to broadcast
        if self.__announcement:
            action.append(self.__shout())
        else:
            # check if anything to speak to agents
            if self.__this_message:
                agent_id, message = self.__this_message
                action.append(self.__whisper(agent_id, message))

        return action

    def decide(self) -> Iterable[VWAction]:
        if self.__stage == 2:
            return self.__help_clean()
        elif self.__stage == 1:
            return self.__explore()
        elif self.__stage == 0:
            return self.__go_bottom_right()

        return [VWIdleAction()]


class CleanerMind(VWActorMindSurrogate):
    def __init__(self) -> None:
        super(CleanerMind, self).__init__()

        self.__master_id: str = ""

        self.__should_take_roll: bool = False
        self.__coord_to_go: VWCoord = VWCoord(-1, -1)
        self.__direction_to_go: VWOrientation = VWOrientation.north

        self.__should_clean: bool = False
        self.__coords_to_clean: list[VWCoord] = []

    def __listen_for_command(self) -> None:
        for m in self.get_latest_received_messages():
            message_content: dict[str, list[str]] = json.loads(str(m.get_content()))
            if message_content.get("command") and message_content["command"]:
                self.__understand_command(m)

    def __understand_command(self, m: BccMessage) -> None:
        message_content: dict[str, list[str]] = json.loads(str(m.get_content()))
        if message_content["command"][0] == "rollcall":
            self.__master_id = m.get_sender_id()
            self.__should_take_roll = True

        if message_content["command"][0] == "getout":
            goto: list[str] = message_content["goto"]
            x, y = goto[0].split(",")
            self.__coord_to_go = VWCoord(int(x), int(y))

        if message_content["command"][0] == "clean":
            colour: str = str(self.get_own_colour())
            coords_list: list[str] = message_content[colour]
            self.__save_coords(coords_list)

        if message_content["command"][0] == "ignore":
            self.__ignore_coord(message_content["coord"][0])

    def __ignore_coord(self, coord: str) -> None:
        x, y = coord.split(",")
        coord_to_ignore = VWCoord(int(x), int(y))
        if coord_to_ignore == self.__coord_to_go:
            self.__coord_to_go = VWCoord(-1, -1)
        if coord_to_ignore in self.__coords_to_clean:
            self.__coords_to_clean.remove(coord_to_ignore)

    def __save_coords(self, coords_list: list[str]) -> None:
        for coord in coords_list:
            x, y = coord.split(",")
            dirt: VWCoord = VWCoord(int(x), int(y))
            if dirt not in self.__coords_to_clean:
                self.__coords_to_clean.append(dirt)

    def __calc_direction_to_go(self) -> VWOrientation:
        now_coord: VWCoord = self.get_own_position()
        now_orientation: VWOrientation = self.get_own_orientation()
        delta_x: int = self.__coord_to_go.get_x() - now_coord.get_x()
        delta_y: int = self.__coord_to_go.get_y() - now_coord.get_y()

        # if delta_x=0,g o N/S based on delta_y
        if delta_x == 0:
            return VWOrientation.north if delta_y < 0 else VWOrientation.south
        # if delta_y=0, go E/W based on delta_x
        elif delta_y == 0:
            return VWOrientation.west if delta_x < 0 else VWOrientation.east

        if (
            now_orientation == VWOrientation.north
            or now_orientation == VWOrientation.south
        ):
            return VWOrientation.north if delta_y < 0 else VWOrientation.south
        else:
            return VWOrientation.west if delta_x < 0 else VWOrientation.east

    def __get_coord_distance(
        self,
        agent_coord: VWCoord,
        target_coord: VWCoord,
    ) -> float:
        # use pythagorean thm to get distance between agent and target cell
        delta_x: int = agent_coord.get_x() - target_coord.get_x()
        delta_y: int = agent_coord.get_y() - target_coord.get_y()
        distance = math.sqrt(delta_x**2 + delta_y**2)
        return distance

    def __get_nearest_coord(self) -> VWCoord:
        nearest_coord: VWCoord = VWCoord(-1, -1)
        nearest_distance: float = math.inf

        agent_coord = self.get_own_position()

        for dirt_coord in self.__coords_to_clean:
            dirt_distance: float = self.__get_coord_distance(agent_coord, dirt_coord)
            if dirt_distance < nearest_distance:
                nearest_distance = dirt_distance
                nearest_coord = dirt_coord

        return nearest_coord

    def __find_coord_to_go(self) -> None:
        self.__coord_to_go = self.__get_nearest_coord()

    def revise(self) -> None:
        self.__should_clean = False
        self.__should_take_roll = False

        # listen for command from master every time
        self.__listen_for_command()

        # if not yet arrived at target coordinate
        if self.get_own_position() != self.__coord_to_go:
            # if target coordinate is invalid, find somewhere to go
            if self.__coord_to_go == VWCoord(-1, -1):
                self.__find_coord_to_go()
            # then, find which direction to go
            self.__direction_to_go = self.__calc_direction_to_go()
        # if arrived at target coordinate
        else:
            # first check if target coordinate has dirt, if so, it needs to be cleaned
            if self.get_latest_observation().get_center().or_else_raise().has_dirt():
                self.__should_clean = True
            # if no dirt, remove current coordinate from cleaning list,
            # tell master this place is cleaned, then
            # find another place to go
            else:
                if self.get_own_position() in self.__coords_to_clean:
                    self.__coords_to_clean.remove(self.get_own_position())
                self.__should_take_roll = True
                self.__should_clean = False
                self.__find_coord_to_go()

        # print(f"{self.get_own_colour()} coord to go:{self.__coord_to_go}")

        print(
            f"{self.get_own_colour()} at {self.get_own_position()} going {self.__coord_to_go}, now cleaning, should clean={self.__should_clean}"
        )
        output: str = f"{self.get_own_colour()} "
        for c in self.__coords_to_clean:
            output += str(c)
        print(output)

    def __go_towards(self, orientation: VWOrientation):
        if self.get_own_appearance().is_facing(orientation):
            return [VWMoveAction()]
        elif self.get_own_appearance().is_facing(orientation.get_left()):
            return [VWTurnAction(VWDirection.right)]
        else:
            return [VWTurnAction(VWDirection.left)]

    def __goto_coord(self, coord: VWCoord) -> Iterable[VWAction]:
        if self.get_own_position() != coord:
            return self.__go_towards(self.__direction_to_go)

        # if own position is already at coord, do nothing
        return [VWIdleAction()]

    def __take_roll(self) -> Iterable[VWAction]:
        position = self.get_own_position()
        about_me: dict[str, str] = {
            "id": self.get_own_id(),
            "colour": str(self.get_own_colour()),
            "coord": f"{position.get_x()},{position.get_y()}",
        }
        return [
            VWSpeakAction(
                message=json.dumps(about_me),
                recipients=[self.__master_id],
                sender_id=self.get_own_id(),
            )
        ]

    def __clean(self) -> Iterable[VWAction]:
        return [VWCleanAction()]

    def decide(self) -> Iterable[VWAction]:
        if self.__should_take_roll and self.__master_id:
            return self.__take_roll()

        if self.get_own_position() == self.__coord_to_go and self.__should_clean:
            return self.__clean()

        if self.__coord_to_go != VWCoord(-1, -1):
            return self.__goto_coord(self.__coord_to_go)

        return [VWIdleAction()]


if __name__ == "__main__":
    run(
        white_mind=ZigZagMind(),
        green_mind=CleanerMind(),
        orange_mind=CleanerMind(),
        efforts=VWActionEffort.REASONABLE_EFFORTS,
        skip=True,
        speed=0.2,
    )
