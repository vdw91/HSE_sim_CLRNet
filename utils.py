# ==================================================================== #
# Copyright (C) 2023 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #

import os
from pathlib import Path


class Pita_Util():
    def __init__(self, root_pth) -> None:
        self.root_pth = root_pth

    # Utils
    def create_folder(self, pth, verbose=False):
        if (Path(pth).exists()):
            pass
        else:
            os.mkdir(pth)
            os.mkdir(pth + '/background')
            os.mkdir(pth + '/frame')
            os.mkdir(pth + '/laneroi')
            os.mkdir(pth + '/segment')

        if (verbose == True):
            print('Save folder is generated!')

    def get_list_of_file_in_a_path(self, fol_path):

        file_list = os.listdir(fol_path)

        # print(file_list)

        return file_list