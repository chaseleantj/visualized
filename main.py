import os
from tkinter import *
from math import sin, cos, tan, asin, acos, atan, sqrt, exp, log
from numpy import sign, array, matmul, linalg, transpose, ndarray, array_equal
from pygame import mixer

SIZE = 600          # size of the grid in pixels
DIV = 40            # number of gridlines shown on screen
STEP = 5            # default distance between gridline labels
ZOOM = 1            # zoom factor
DURATION = 2000     # default length of animation
FR = 40             # number of frames per animation
POINTER = 0         # pointer indicating the current history position (for undo/redo transformations)

pi = 3.1415926535
e = 2.7182818284
phi = 1.6180339887

# the gray grid in the background that is not affected by linear transformations
class Grid:
    def __init__(self, master=None):
        self.master = master
        self.step = STEP

        self.x_lines = [self.master.create_line(i * SIZE / DIV, 0, i * SIZE / DIV, SIZE) for i in range(DIV)]
        self.y_lines = [self.master.create_line(0, i * SIZE / DIV, SIZE, i * SIZE / DIV) for i in range(DIV)]

        self.configure_lines()
        self.update()

    # updates the labels when zooming in or out
    def update(self):

        try:
            for label in self.x_labels:
                self.master.delete(label)
            for label in self.y_labels:
                self.master.delete(label)
        except AttributeError as e:
            print("[Grid section]", e)

        self.x_labels = [self.master.create_text(cts(i / ZOOM, "x") - 10, cts(0, "y") + 15, text=str(int(i / ZOOM))
        if abs(i / ZOOM) >= 1 or abs(i / ZOOM) == 0 else str(i / ZOOM), fill="gray50",
                                                 font=('Consolas', 12)) for i in
                         range(int(-DIV / 2), int(DIV / 2), self.step)]
        # does not generate a label for '0' on the y-axis to prevent two '0's being shown
        self.y_labels = [self.master.create_text(cts(0, "x") - 20, cts(i / ZOOM, "y"), text=str(int(i / ZOOM))
        if abs(i / ZOOM) >= 1 else str(i / ZOOM), fill="gray50", font=('Consolas', 12)) for i in
                         range(int(-DIV / 2), int(DIV / 2), self.step) if i != 0]

    # configures the color and thickness of the gridlines
    def configure_lines(self):
        colors = ("gray25", "gray15", "gray10")
        widths = (3, 2, 1)
        for i in range(DIV):
            if i % (DIV / 2) == 0:
                self.master.itemconfig(self.x_lines[i], fill=colors[0], width=widths[0], tags=("gridline", "top"))
                self.master.itemconfig(self.y_lines[i], fill=colors[0], width=widths[0], tags=("gridline", "top"))
            elif i % self.step == 0:
                self.master.itemconfig(self.x_lines[i], fill=colors[1], width=widths[1], tags=("gridline", "middle"))
                self.master.itemconfig(self.y_lines[i], fill=colors[1], width=widths[1], tags=("gridline", "middle"))
            else:
                self.master.itemconfig(self.x_lines[i], fill=colors[2], width=widths[2], tags=("gridline", "bottom"))
                self.master.itemconfig(self.y_lines[i], fill=colors[2], width=widths[2], tags=("gridline", "bottom"))

            # ensures that thicker gridlines always appear on top of thinner gridlines, but gridlines lie below
            # all other canvas objects
            self.master.tag_raise("middle")
            self.master.tag_raise("top")
            self.master.tag_lower("gridline")

# the blue grid in the background that is affected by linear transformations
class Mesh:
    def __init__(self, master=None):
        self.master = master
        self.step = STEP
        self.lock = False   # when the mesh is locked, zooming in/out will not affect it
        self.history = []

        self.refresh()
        self.configure_lines()

    def configure_lines(self):
        colors = ("#007777", "#005555", "#004444")
        widths = (3, 2, 1)
        for i in range(1, 3 * DIV):
            if i % (3 * DIV / 2) == 0:
                self.master.itemconfig(self.x_lines[i], fill=colors[0], width=widths[0], tags=("gridline", "top"))
                self.master.itemconfig(self.y_lines[i], fill=colors[0], width=widths[0], tags=("gridline", "top"))
            elif i % self.step == 0:
                self.master.itemconfig(self.x_lines[i], fill=colors[1], width=widths[1], tags=("gridline", "middle"))
                self.master.itemconfig(self.y_lines[i], fill=colors[1], width=widths[1], tags=("gridline", "middle"))
            else:
                self.master.itemconfig(self.x_lines[i], fill=colors[2], width=widths[2], tags=("gridline", "bottom"))
                self.master.itemconfig(self.y_lines[i], fill=colors[2], width=widths[2], tags=("gridline", "bottom"))

            self.master.tag_raise("middle")
            self.master.tag_raise("top")
            self.master.tag_lower("gridline")

    def update(self):
        for line in self.x_lines:
            self.master.delete(line)
        for line in self.y_lines:
            self.master.delete(line)
        self.x_lines = [self.master.create_line(cts(line[0][0], "x"), cts(line[0][1], "y"), cts(line[1][0], "x"),
                                                cts(line[1][1], "y")) for line in self.x_coords]
        self.y_lines = [self.master.create_line(cts(line[0][0], "x"), cts(line[0][1], "y"), cts(line[1][0], "x"),
                                                cts(line[1][1], "y")) for line in self.y_coords]
        self.configure_lines()

    def transform(self, matrices, angles, transform_no=0):
        # locks the mesh in place so that zooming in/out will not change its size
        self.lock = True

        # calculates the destination of the endpoints of each mesh line
        x_destination = [[matmul(matrices[transform_no], loc) for loc in line] for line in self.x_coords]
        y_destination = [[matmul(matrices[transform_no], loc) for loc in line] for line in self.y_coords]

        # finds the change in distance between the initial point and the destination
        delta_x = [[(f_loc - i_loc) / FR for i_loc, f_loc in zip(i_line, f_line)] for i_line, f_line in
                   zip(self.x_coords, x_destination)]
        delta_y = [[(f_loc - i_loc) / FR for i_loc, f_loc in zip(i_line, f_line)] for i_line, f_line in
                   zip(self.y_coords, y_destination)]

        # forms a rotation matrix using the angle between the initial line and final line (only used when the linear
        # transformation is a pure rotation
        rotation_matrix = array([[cos(angles[transform_no] / FR), - sin(angles[transform_no] / FR)],
                                 [sin(angles[transform_no] / FR), cos(angles[transform_no] / FR)]])
        self.animate(delta_x, delta_y, x_destination, y_destination, rotation_matrix, 0)

        # if the linear transformation is a composition of matrices, call the transform function recursively until
        # all transformations are applied
        if transform_no < len(matrices) - 1:
            root.after(DURATION - 100, lambda: self.transform(matrices, angles, transform_no + 1))

    # plays the animation by moving each line from its initial position to the final position in increments
    def animate(self, delta_x, delta_y, x_destination, y_destination, rotation_matrix, frame=0):
        if frame < FR - 1:
            if acos(rotation_matrix[0][0]) == 0:
                self.x_coords = [[i_loc + dx for i_loc, dx in zip(i_line, d_line)] for i_line, d_line in
                                 zip(self.x_coords, delta_x)]
                self.y_coords = [[i_loc + dy for i_loc, dy in zip(i_line, d_line)] for i_line, d_line in
                                 zip(self.y_coords, delta_y)]
            else:
                self.x_coords = [[matmul(rotation_matrix, loc) for loc in line] for line in self.x_coords]
                self.y_coords = [[matmul(rotation_matrix, loc) for loc in line] for line in self.y_coords]
            self.update()
            self.master.after(int(DURATION / 100),
                              lambda: self.animate(delta_x, delta_y, x_destination, y_destination, rotation_matrix,
                                                   frame + 1))
        else:
            self.x_coords = x_destination
            self.y_coords = y_destination
            self.update()
            return

    # records the position of every mesh line and writes it into memory, allowing the user to undo and redo the actions
    def record(self, time=0):
        if time == 0:
            self.history.append(array([self.x_coords, self.y_coords]))
        else:
            self.master.after(time, lambda: self.history.append(array([self.x_coords, self.y_coords])))

    # returns the mesh to its original position
    def refresh(self):
        try:
            for line in self.x_lines:
                self.master.delete(line)
            for line in self.y_lines:
                self.master.delete(line)
        except AttributeError as e:
            print("[Mesh section]", e)

        self.x_lines = [self.master.create_line(i * SIZE / DIV, - SIZE, i * SIZE / DIV, 2 * SIZE) for i in
                        range(-DIV, 2 * DIV, 1)]
        self.y_lines = [self.master.create_line(- SIZE, i * SIZE / DIV, 2 * SIZE, i * SIZE / DIV) for i in
                        range(-DIV, 2 * DIV, 1)]

        self.x_coords = [self.master.coords(line) for line in self.x_lines]
        self.x_coords = [[stc(loc, "x") for loc in line] for line in self.x_coords]
        self.x_coords = [[line[0:2], line[2:4]] for line in self.x_coords]

        self.y_coords = [self.master.coords(line) for line in self.y_lines]
        self.y_coords = [[stc(loc, "y") for loc in line] for line in self.y_coords]
        self.y_coords = [[line[0:2], line[2:4]] for line in self.y_coords]

        self.configure_lines()
        self.history.append(array([self.x_coords, self.y_coords]))


class Vector:
    def __init__(self, name, no, master=None):
        colors = ("#FFFF00", "#00FFFF", "#FF00FF")
        self.master = master
        self.name = name
        self.color = colors[no]
        self.origin = SIZE / 2
        self.name = StringVar()
        self.name.set(f"V{no + 1}")
        self.input = StringVar()
        self.input.set("[0; 0]")
        self.status = BooleanVar()          # whether the vector is visible on screen
        self.status.set(False)
        self.value = array([0, 0])          # the actual position of the vector, (x, y), stored as a numpy array
        self.history = [array([0, 0])]

        self.button = Button(vector_frame, image=hide_image, bg="black", activebackground="black", border=0,
                             command=lambda: (self.toggle(), click_sound()))
        self.name_label = Entry(vector_frame, width=7, textvariable=self.name, font=('Corbel', 16, 'bold'),
                                fg=self.color, bg="black", insertbackground="white", relief=FLAT, justify=RIGHT)
        self.equals_label = Label(vector_frame, text="=", font=('Corbel', 20), fg="white", bg="black")
        self.input_label = Entry(vector_frame, width=22, textvariable=self.input, font=('Consolas', 14),
                                 justify="center", fg="white", bg="gray10", insertbackground="white", relief=FLAT)

        # binding the input entry to every key, so that whenever the user changes any input, the validity of the input
        # is automatically detected; root.after(10) to ensure the user input is registered BEFORE it is checked
        self.input_label.bind('<Key>', lambda x: root.after(10, lambda: (
        self.update(), self.toggle() if not self.status.get() else None)))

        self.name_label.grid(row=no + 4, column=0, pady=2)
        self.equals_label.grid(row=no + 4, column=1, padx=10, pady=2)
        self.input_label.grid(row=no + 4, column=2, pady=2)
        self.button.grid(row=no + 4, column=3, padx=(10, 30), pady=2)

        # changing default settings for the first vector
        if no == 0:
            self.status.set(True)
            self.button.config(image=show_image)
            self.input.set("[10; 10]")
            self.history[0] = array([10, 10])

        self.update()

    # toggles the visibility of the vector
    def toggle(self):
        if self.status.get():
            self.status.set(False)
            self.button.config(image=hide_image)
            try:
                self.master.itemconfig(self.arrow, state=HIDDEN)
                self.master.itemconfig(self.label, state=HIDDEN)
            except AttributeError:
                print("[Vector section] Nonexistent arrow")
        else:
            self.status.set(True)
            self.button.config(image=show_image)
            try:
                self.master.itemconfig(self.arrow, state=NORMAL)
                self.master.itemconfig(self.label, state=NORMAL)
            except AttributeError:
                print("[Vector section] Nonexistent arrow")
            self.update()

    # updates the position of the vector based on the input entry
    def update(self):
        try:
            if not (self.input.get()[0] == "[" and self.input.get()[-1] == "]"):
                self.input_label.configure(bg="#440000")
                return
            user_input = self.input.get()[1:-1].split(";")
            x = eval(user_input[0].strip(" "))
            y = eval(user_input[1].strip(" "))
            self.value = [x, y]
            self.input_label.configure(bg="#002222")
            self.movement(x, y)
        except (TypeError, SyntaxError, NameError, IndexError) as e:
            self.input_label.configure(bg="#440000")
            print("[Vector section]", e)

    # moves the vector to the specified location instantly
    def movement(self, x_prime, y_prime):
        try:
            self.master.delete(self.label)
            self.master.delete(self.arrow)
        except AttributeError as e:
            print("[Vector section]", e)
        if self.status.get():
            self.arrow = self.master.create_line(self.origin, self.origin, cts(x_prime, "x"), cts(y_prime, "y"),
                                                 fill=self.color, width=3, arrow=LAST, arrowshape=(18.75, 18, 5.25),
                                                 tags="vector")
            self.gen_label(x_prime, y_prime)

    def transform(self, matrices, angles, transform_no=0):
        destination = matmul(matrices[transform_no], self.value)
        destination = array([row.round(decimals=5) for row in destination])
        [dx, dy] = [(destination[i] - self.value[i]) / FR for i in range(2)]
        self.animate(dx, dy, angles[transform_no] / FR, destination, 0)
        if transform_no < len(matrices) - 1:
            root.after(DURATION - 100, lambda: self.transform(matrices, angles, transform_no + 1))

    def animate(self, dx, dy, angle, destination, frame=0):
        if frame < FR - 1:
            if angle == 0:
                self.input.set(f"[{round(self.value[0] + dx, 5)}; {round(self.value[1] + dy, 5)}]")
            else:
                self.input.set(f"[{round(cos(angle) * self.value[0] - sin(angle) * self.value[1], 5)}; "
                               f"{round(sin(angle) * self.value[0] + cos(angle) * self.value[1], 5)}]")
            self.update()
            self.master.after(int(DURATION / 100), lambda: self.animate(dx, dy, angle, destination, frame + 1))
        else:
            if destination[0] == int(destination[0]) and destination[1] == int(destination[1]):
                self.input.set(f"[{int(destination[0])}; {int(destination[1])}]")
            else:
                # rounding might cause loss of precision
                self.input.set(f"[{round(destination[0], 5)}; {round(destination[1], 5)}]")
            self.update()
            return

    # generates a label (x, y) indicating the terminal point of the vector
    def gen_label(self, x, y):
        if y != 0:
            self.label = self.master.create_text(cts(x, "x") + 15 * sign(x),
                                                 cts(y, "y") - 15 * sign(y),
                                                 text=f"({round(x, 3)}, {round(y, 3)})",
                                                 fill=self.color,
                                                 font=("Consolas", 12), tags="vector_label")
        else:
            self.label = self.master.create_text(cts(x, "x"),
                                                 cts(y + 2 / ZOOM * DIV / 60, "y") - 15 * sign(y),
                                                 text=f"({round(x, 3)}, {round(y, 3)})",
                                                 fill=self.color,
                                                 font=("Consolas", 12), tags="vector_label")

    def record(self, time=0):
        if time == 0:
            self.history.append(array(self.value))
        else:
            self.master.after(time, lambda: self.history.append(array(self.value)))


class Matrix:
    def __init__(self, name, no, master=None):
        self.master = master
        self.name = name
        self.origin = SIZE / 2
        self.name = StringVar()
        self.name.set(f"M{no + 1}")
        self.input = StringVar()
        self.input.set("[1, 0; 0, 1]")
        self.status = BooleanVar()
        self.status.set(True)
        self.value = array([[1, 0], [0, 1]])

        self.name_label = Entry(matrix_frame, width=7, textvariable=self.name, font=('Corbel', 16, 'bold'),
                                fg="white", bg="black", insertbackground="white", relief=FLAT, justify=RIGHT)
        self.equals_label = Label(matrix_frame, text="=", font=('Corbel', 20), fg="white", bg="black")
        self.input_label = Entry(matrix_frame, width=25, textvariable=self.input, font=('Consolas', 14),
                                 justify="center", fg="white", bg="gray10", insertbackground="white", relief=FLAT)
        self.input_label.bind('<Key>', lambda x: root.after(10, lambda: self.verify()))

        self.name_label.grid(row=no + 1, column=0, pady=2)
        self.equals_label.grid(row=no + 1, column=1, padx=10, pady=2)
        self.input_label.grid(row=no + 1, column=2, padx=(0, 40), pady=2)

        self.verify()

    # this function checks if the input matrix is valid; it works by doing the following:
    # try:
    #   check if the input is a standard matrix
    # except:
    #   try:
    #       check if the input is a supported function
    #   except:
    #       turn the input box red
    def verify(self):
        try:
            user_input = self.input.get()[1:-1].split(";")
            user_input = [row.split(',') for row in user_input]
            user_input = [[eval(element.strip(" ")) for element in row] for row in user_input]
            user_input = array(user_input)

            # testing if input matrix can be multiplied by a 2x1 column vector
            test_vector = array([1, 1])
            matmul(user_input[1], test_vector)

            self.status.set(True)
            self.value = user_input
            self.input_label.configure(bg="#002222")

        except (TypeError, SyntaxError, ValueError, NameError, IndexError) as e:
            try:
                user_input = eval(self.input.get())

                # testing if input matrix can be multiplied by a 2x1 column vector
                test_vector = array([1, 1])
                matmul(user_input[1], test_vector)

                self.status.set(True)
                self.value = user_input
                self.input_label.configure(bg="#002222")
            except (TypeError, SyntaxError, ValueError, NameError, IndexError) as e:
                self.status.set(False)
                self.input_label.configure(bg="#440000")
                print("[Matrix section]", "Invalid input")


# converts coordinates on screen (in pixels) to displayed coordinates (in units)
def stc(screen_x, xy):
    x0 = SIZE / 2
    scale_factor = SIZE / DIV * ZOOM
    if xy == "x":
        coords_x = (screen_x - x0) / scale_factor
    else:
        coords_x = - (screen_x - x0) / scale_factor
    return coords_x


# converts displayed coordinates to coordinates in screen
def cts(coords_x, xy):
    x0 = SIZE / 2
    scale_factor = SIZE / DIV * ZOOM
    if xy == "x":
        screen_x = coords_x * scale_factor + x0
    else:
        screen_x = - coords_x * scale_factor + x0
    return screen_x


# identity matrix
def identity():
    return array([[1, 0], [0, 1]])


# rotate by the specified number of radians
def rotate(angle):
    return array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])


# rotate by the specified number of degrees
def rotatedeg(angle):
    angle *= pi / 180
    return array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])


# enlarge/shrink by the specified scale factor
def scale(factor):
    return array([[factor, 0], [0, factor]])


# reflect with respect to the specified axis
def reflect(axis):
    if axis == 'x':
        return array([[1, 0], [0, -1]])
    elif axis == 'y':
        return array([[-1, 0], [0, 1]])
    # enables lines like "y = 2*x" to be chosen as the axis of reflection
    elif axis[-1] == 'x':
        try:
            m = eval(axis.split("*")[0].strip(" "))
            a = (1 - m ** 2) / (m ** 2 + 1)
            b = 2 * m / (m ** 2 + 1)
            return array([[a, b], [b, -a]])
        except TypeError as e:
            print(e)
    else:
        return False


# project on the specified axis
def project(axis):
    if axis == 'x':
        return array([[1, 0], [0, 0]])
    elif axis == 'y':
        return array([[0, 0], [0, 1]])
    # enables lines like "y = 2*x" to be chosen as the axis of reflection
    elif axis[-1] == 'x':
        try:
            m = eval(axis.split("*")[0].strip(" "))
            a = m ** 2 + 1
            return array([[1 / a, m / a], [m / a, m ** 2 / a]])
        except TypeError as e:
            print(e)
    else:
        return False


# shear with respect to the specified axis, with the specified shear factor
def shear(axis, factor=1):
    if axis == 'x':
        return array([[1, factor], [0, 1]])
    elif axis == 'y':
        return array([[1, 0], [factor, 1]])
    else:
        return False


# locates the required file, enabling files to be bundled up in a standalone executable
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    def zoom(type):
        click_sound()
        global ZOOM

        # the threshold is a value used to prevent weird scales such as 4, 8, 12...
        # when this happens the zoom factor is changed, adjusting the scale to a suitable value like 5, 10, 15...
        threshold = "{:e}".format(ZOOM)[0:3]
        if type == "OUT":
            if threshold == "2.5":
                ZOOM = ZOOM * 4 / 10
            else:
                ZOOM = ZOOM / 2
        elif type == "IN":
            if threshold == "1.0":
                ZOOM = ZOOM * 10 / 4
            else:
                ZOOM = ZOOM * 2
        grid.update()
        if transforming.get():  # rescales the mesh during a linear transformation when user zooms in/out
            mesh.update()
        else:
            # rescales the mesh when a linear transformation has been completed, so that user can better
            # view the effects of the linear transformation
            if mesh.lock:
                mesh.update()
            else:
                mesh.refresh()  # otherwise, resize the mesh such that it covers the entire screen
        for vector in vector_list:
            vector.update()

    # checks if the names of the matrices entered are correct
    def verify_command():
        user_input = command.get().split("*")
        user_input = [word.strip(" ") for word in user_input]
        for matrix in user_input:
            if matrix not in [matrix.name.get().strip(" ") for matrix in matrix_list]:
                transform_entry.configure(bg="#440000")
                transform_button.configure(state=DISABLED)
                return False
        transform_entry.configure(bg="#002222")
        transform_button.configure(state=NORMAL)
        return True


    def apply_transform(*args):
        if not verify_command():
            return

        # parses the user input
        transform_entry.unbind('<Return>')
        user_input = command.get().split("*")
        user_input = [word.strip(" ") for word in user_input]

        # generates a list of the matrices that the user is referring to, in the correct order
        matrices = []
        for word in reversed(user_input):
            for matrix in matrix_list:
                if matrix.name.get().strip(" ") == word:
                    matrices.append(matrix)
                    break

        for matrix in matrices:
            if not matrix.status.get():
                transform_entry.configure(bg="#440000")
                transform_button.configure(state=DISABLED)
                command.set("\tOne of the matrices is invalid.")
                transform_entry.bind('<Return>', apply_transform)
                root.after(3000, lambda: (command.set(""), transform_entry.configure(bg="#002222")))
                return

        # obtains the value of the specified matrices, in the correct order
        transformation = [array(matrix.value) for matrix in matrices]

        # checks if the matrices are purely rotation matrices, then calculating their angle
        # the angle is set to 0 if the matrix is not a pure rotation matrix
        angles = []
        for matrix in transformation:
            try:
                if ndarray.all(linalg.inv(matrix).round(decimals=5) == transpose(matrix).round(decimals=5)) and round(
                        linalg.det(matrix), 5) == 1:
                    if asin(matrix[1][0]) >= 0:
                        angles.append(acos(matrix[0][0]))
                    else:
                        angles.append(2 * pi - acos(matrix[0][0]))
                else:
                    angles.append(0)
            except linalg.LinAlgError:
                print("Singular matrix")
                angles.append(0)

        global POINTER

        # if the user applies transformation A and then B
        # then presses UNDO, so that the screen reverts from transformation B back to A
        # the user then applies transformation C
        # this line of code erases the memory of transformation B
        for vector in vector_list:
            if not array_equal(vector.history[POINTER], vector.value):
                POINTER += 1
                for v in vector_list:
                    v.record()
                mesh.record()
                break

        POINTER += 1

        # records the location of every object in canvas into their history attributes, enabling UNDO and REDO
        if len(mesh.history) - 1 > POINTER:
            del mesh.history[POINTER:]
            for vector in vector_list:
                del vector.history[POINTER:]
            redo_button.configure(state=DISABLED)

        # disables buttons during the transformation
        refresh_mesh_button.configure(state=DISABLED)
        transform_button.configure(state=DISABLED)
        undo_button.configure(state=DISABLED)
        transforming.set(True)
        sound_system(transformation)

        for vector in vector_list:
            if vector.status.get():
                vector.transform(transformation, angles)
            vector.record(DURATION * len(transformation))

        mesh.transform(transformation, angles)
        mesh.record(DURATION * len(transformation))

        # enables buttons after transformation is complete
        root.after(DURATION * len(transformation), lambda: (refresh_mesh_button.configure(state=NORMAL),
                                                            transform_button.configure(state=NORMAL),
                                                            undo_button.configure(state=NORMAL),
                                                            transforming.set(False),
                                                            transform_entry.bind('<Return>', apply_transform)))


    def music_config():
        if music.get():
            music_button.configure(image=music_off_image)
            music.set(False)
            music_sound.set_volume(0)
        else:
            music_button.configure(image=music_on_image)
            music.set(True)
            music_sound.set_volume(0.2)
        # remembers the user's settings so that the next time they open the program their preferences remain saved
        with open(resource_path("preferences.txt"), "w") as file:
            file.write(str(music.get()) + "," + str(audio.get()))

    def audio_config():
        if audio.get():
            audio_button.configure(image=audio_off_image)
            audio.set(False)
        else:
            audio_button.configure(image=audio_on_image)
            audio.set(True)
        # remembers the user's settings so that the next time they open the program their preferences remain saved
        with open(resource_path("preferences.txt"), "w") as file:
            file.write(str(music.get()) + "," + str(audio.get()))

    def click_sound():
        if audio.get():
            button_sound.play()
        else:
            return
    
    # determines which sound effect to play based on the determinant of the matrix 
    def sound_system(matrices, count=0):
        if audio.get() and count < len(matrices):
            if linalg.det(matrices[count]) < 0:
                effect1_sound.play()
            elif linalg.det(matrices[count]) == 0:
                effect2_sound.play()
            else:
                effect3_sound.play()
        root.after(DURATION, lambda: sound_system(matrices, count + 1))


    def refresh_mesh():
        click_sound()
        refresh_mesh_button.configure(state=DISABLED)

        global POINTER
        POINTER += 1

        mesh.lock = False
        mesh.refresh()
        for vector in vector_list:
            vector.record()


    def undo():
        click_sound()
        global POINTER
        if POINTER > 0:
            POINTER -= 1
            if POINTER == 0:
                undo_button.configure(state=DISABLED)
            for vector in vector_list:
                vector.input.set(f"[{vector.history[POINTER][0]}; {vector.history[POINTER][1]}]")
                vector.update()
            mesh.x_coords, mesh.y_coords = mesh.history[POINTER][0], mesh.history[POINTER][1]
            mesh.update()
            redo_button.configure(state=NORMAL)
        else:
            undo_button.configure(state=DISABLED)


    def redo():
        click_sound()
        global POINTER
        if POINTER < len(mesh.history) - 1:
            POINTER += 1
            if POINTER == len(mesh.history) - 1:
                redo_button.configure(state=DISABLED)
            for vector in vector_list:
                vector.input.set(f"[{vector.history[POINTER][0]}; {vector.history[POINTER][1]}]")
                vector.update()
            mesh.x_coords, mesh.y_coords = mesh.history[POINTER][0], mesh.history[POINTER][1]
            mesh.update()
            undo_button.configure(state=NORMAL)
        else:
            redo_button.configure(state=DISABLED)

    # speeds up the animation of linear transformations
    def fast_forward():
        click_sound()
        global DURATION
        if DURATION == 4000:
            fast_forward_button.configure(image=speed2_image)
            DURATION = 2000
        elif DURATION == 2000:
            fast_forward_button.configure(image=speed3_image)
            DURATION = 1000
        else:
            fast_forward_button.configure(image=speed1_image)
            DURATION = 4000


    def glow(e):
        transform_button.configure(image=go_light_image)


    def dim(e):
        transform_button.configure(image=go_image)

    # generates a new window showing the instructions
    def instructions():
        click_sound()
        root_2 = Toplevel()
        root_2.title("Instructions")
        Label(root_2, image=instructions_image).pack()


    root = Tk()
    base = Frame(root)  # a frame that ensures the widgets remain in the middle of the screen even in fullscreen mode
    left_frame = Frame(base)
    vector_frame = Frame(base)
    matrix_frame = Frame(base)
    transform_frame = Frame(base)
    canvas = Canvas(left_frame)

    # ensures that the Tkinter window appears in the middle of the screen every time
    root.title("2-D Linear transformations visualized")
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry('+%d+%d' % (sw / 2 - 550, sh / 2 - 350))
    base.pack(fill="none", expand=True)

    root.configure(bg="black")
    base.configure(bg="black")
    left_frame.configure(bg="black")
    vector_frame.configure(bg="black")
    matrix_frame.configure(bg="black")
    transform_frame.configure(bg="black")
    canvas.configure(bg="black", width=SIZE, height=SIZE)

    canvas.grid(row=0, column=0)
    left_frame.grid(row=0, rowspan=3, column=0, padx=(20, 0), pady=20)
    vector_frame.grid(row=0, column=1, padx=5)
    matrix_frame.grid(row=1, column=1, padx=5)
    transform_frame.grid(row=2, column=1, padx=5)

    command = StringVar()
    transforming = BooleanVar()
    transforming.set(False)
    music = BooleanVar()
    music.set(True)
    audio = BooleanVar()
    audio.set(True)

    logo_image = PhotoImage(file=resource_path("Files\\Icons\\logo.png"))
    zoom_in_image = PhotoImage(file=resource_path("Files\\Icons\\zoom_in.png"))
    zoom_out_image = PhotoImage(file=resource_path("Files\\Icons\\zoom_out.png"))
    music_on_image = PhotoImage(file=resource_path("Files\\Icons\\music_on.png"))
    music_off_image = PhotoImage(file=resource_path("Files\\Icons\\music_off.png"))
    audio_on_image = PhotoImage(file=resource_path("Files\\Icons\\audio_on.png"))
    audio_off_image = PhotoImage(file=resource_path("Files\\Icons\\audio_off.png"))
    show_image = PhotoImage(file=resource_path("Files\\Icons\\show.png"))
    hide_image = PhotoImage(file=resource_path("Files\\Icons\\hide.png"))
    go_image = PhotoImage(file=resource_path("Files\\Icons\\go.png"))
    go_light_image = PhotoImage(file=resource_path("Files\\Icons\\go_light.png"))
    help_image = PhotoImage(file=resource_path("Files\\Icons\\help.png"))
    refresh_image = PhotoImage(file=resource_path("Files\\Icons\\refresh.png"))
    undo_image = PhotoImage(file=resource_path("Files\\Icons\\undo.png"))
    redo_image = PhotoImage(file=resource_path("Files\\Icons\\redo.png"))
    speed1_image = PhotoImage(file=resource_path("Files\\Icons\\speed1.png"))
    speed2_image = PhotoImage(file=resource_path("Files\\Icons\\speed2.png"))
    speed3_image = PhotoImage(file=resource_path("Files\\Icons\\speed3.png"))
    instructions_image = PhotoImage(file=resource_path("Files\\Icons\\instructions.png"))

    mixer.init()
    button_sound = mixer.Sound(resource_path("Files\\Audio\\button.mp3"))
    button_sound.set_volume(0.2)
    effect1_sound = mixer.Sound(resource_path("Files\\Audio\\effect1.wav"))
    effect1_sound.set_volume(0.1)
    effect2_sound = mixer.Sound(resource_path("Files\\Audio\\effect2.wav"))
    effect2_sound.set_volume(0.1)
    effect3_sound = mixer.Sound(resource_path("Files\\Audio\\effect3.wav"))
    effect3_sound.set_volume(0.3)
    music_sound = mixer.Sound(resource_path("Files\\Audio\\music.mp3"))
    music_sound.set_volume(0.2)
    music_sound.play(-1)

    logo_label = Label(vector_frame, image=logo_image, background="black")
    zoom_in = Button(vector_frame, image=zoom_in_image, bg="black", activebackground="black", fg="white", border=0,
                     command=lambda: zoom("IN"))
    zoom_out = Button(vector_frame, image=zoom_out_image, bg="black", activebackground="black", fg="white", border=0,
                      command=lambda: zoom("OUT"))
    music_button = Button(vector_frame, image=music_on_image, bg="black", activebackground="black", fg="white",
                          border=0,
                          command=music_config)
    audio_button = Button(vector_frame, image=audio_on_image, bg="black", activebackground="black", fg="white",
                          border=0,
                          command=audio_config)
    instructions_button = Button(vector_frame, text="Click here for the instructions.", bg="black",
                                 activebackground="black", fg="white", border=0, font=('Corbel', 12, 'bold'),
                                 command=instructions)
    instructions_button.bind('<Enter>', lambda x: instructions_button.configure(fg="#00FFFF"))
    instructions_button.bind('<Leave>', lambda x: instructions_button.configure(fg="white"))
    vector_label = Label(vector_frame, text="Initialize vectors", bg="black", fg="white", font=('Corbel', 12, 'bold'),
                         justify=CENTER)
    matrix_label = Label(matrix_frame, text="Initialize matrices", bg="black", fg="white", font=('Corbel', 12, 'bold'),
                         justify=CENTER)

    logo_label.grid(row=0, rowspan=2, column=0, columnspan=4)
    zoom_in.grid(row=0, column=0, padx=(0, 10), sticky="w")
    zoom_out.grid(row=1, column=0, padx=(0, 10), sticky="nw")
    music_button.grid(row=2, column=0, padx=(0, 10), sticky="w")
    audio_button.grid(row=3, column=0, padx=(0, 10), sticky="w")
    instructions_button.grid(row=2, column=0, columnspan=4, pady=10)
    vector_label.grid(row=3, column=0, columnspan=4)
    matrix_label.grid(row=0, column=0, columnspan=3)

    transform_label = Label(transform_frame, text="Enter the linear transformation below."
                                                  "\nIt will be applied to all visible vectors.",
                            bg="black", fg="white", font=('Corbel', 12, 'bold'), justify=CENTER)
    transform_entry = Entry(transform_frame, textvariable=command, width=25, relief=FLAT, bg="#002222", fg="white",
                            insertbackground="white", font=('Corbel', 16, 'bold'), justify="center")
    transform_entry.bind("<Key>", lambda x: root.after(10, verify_command))
    transform_entry.bind("<Return>", apply_transform)
    transform_button = Button(transform_frame, image=go_image, bg="black", fg="black", activebackground="Black",
                              border=0, font=('Arial', 12), command=lambda: (apply_transform(), click_sound()),
                              state=DISABLED)
    transform_button.bind('<Enter>', glow)
    transform_button.bind('<Leave>', dim)
    help_button = Button(transform_frame, image=help_image, bg="black", activebackground="black", border=0,
                         command=instructions)
    refresh_mesh_button = Button(transform_frame, image=refresh_image, bg="black", activebackground="black",
                                 command=refresh_mesh, border=0, state=DISABLED)
    fast_forward_button = Button(transform_frame, image=speed2_image, bg="black", activebackground="black", border=0,
                                 command=fast_forward)
    undo_button = Button(transform_frame, image=undo_image, bg="black", activebackground="black", border=0,
                         command=undo, state=DISABLED)
    redo_button = Button(transform_frame, image=redo_image, bg="black", activebackground="black", border=0,
                         command=redo, state=DISABLED)

    transform_label.grid(row=0, column=0, columnspan=5, pady=5)
    transform_entry.grid(row=1, column=0, columnspan=4, padx=(10, 5), pady=5)
    transform_button.grid(row=1, column=4, pady=5)
    help_button.grid(row=2, column=0, sticky="e")
    refresh_mesh_button.grid(row=2, column=1)
    fast_forward_button.grid(row=2, column=2)
    undo_button.grid(row=2, column=3)
    redo_button.grid(row=2, column=4, sticky="w")

    # initializes the grid, mesh, vectors and matrices together with their inputs
    grid = Grid(canvas)
    mesh = Mesh(canvas)
    vector_list = [Vector(f"V{i}", i, canvas) for i in range(3)]
    matrix_list = [Matrix(f"M{i}", i, canvas) for i in range(3)]

    # stores vector, matrix and transformation entry widgets into a list
    entry_list = []
    for vector in vector_list:
        entry_list.append(vector.input_label)
    for matrix in matrix_list:
        entry_list.append(matrix.input_label)
    entry_list.append(transform_entry)

    # allows the user to move to different entries by pressing the UP and DOWN keys
    for i in range(len(entry_list)):
        if i != 0:
            entry_list[i].bind('<Up>', lambda event, x=i: entry_list[x - 1].focus())
        if i != len(entry_list) - 1:
            entry_list[i].bind('<Down>', lambda event, x=i: entry_list[x + 1].focus())

    # remembers whether the user turned the music/audio on or off the last time they used the program and
    # changes the music/audio settings accordingly
    with open(resource_path("Files\\preferences.txt"), "r") as file:
        preference = file.read().split(",")
        if preference[0] == 'False':
            music_button.configure(image=music_off_image)
            music.set(False)
            music_sound.set_volume(0)
        if preference[1] == 'False':
            audio_button.configure(image=audio_off_image)
            audio.set(False)

    root.mainloop()
