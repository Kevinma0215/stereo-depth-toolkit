""" 建立 board/dictionary（可重用） """
from __future__ import annotations
import cv2

def make_dictionary(name: str):
    aruco = cv2.aruco
    dict_id = getattr(aruco, name)
    return aruco.getPredefinedDictionary(dict_id)

def make_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dict_name: str = "DICT_4X4_50",
):
    aruco = cv2.aruco
    dictionary = make_dictionary(dict_name)
    board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    return board, dictionary
