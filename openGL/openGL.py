# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
import numpy as np
import pywavefront

# global window_height, window_width
window_height, window_width = 400, 600

# global fov, aspect, z_near, z_far
fov, aspect, z_near, z_far = 60.0, window_width / window_height, 1., 100.

# global angle_speed, mouse_speed, move_speed
angle_speed, mouse_speed, move_speed = 5, 5e-3, 5.
angles = np.zeros(2)
# global eye, target, up, MAX_EYE_DIST
eye = np.array([0.0, 0.0, -2.0])
target = np.array([0.0, 0.0, 1.0])
up = np.array([0.0, 1.0, 0.0])
MAX_EYE_DIST = 25.0

global m_mat, v_mat, mv_mat, p_mat, mvp_mat
global color

global loc_m_mat_uc, loc_mv_mat_uc, loc_mvp_mat_uc
global loc_m_mat_ac, loc_mv_mat_ac, loc_mvp_mat_ac
global loc_color

global shader_uniform_color, shader_attribute_color

global meshes

monkey = pywavefront.Wavefront('monkey.obj')

move, left, right, forward, backward, upward, downward = 0, 1, 2, 4, 8, 16, 32


class Matrices:
    @staticmethod
    def translate(to):
        return np.array([
            [1, 0, 0, to[0]],
            [0, 1, 0, to[1]],
            [0, 0, 1, to[2]],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def rotate(by, v):
        s, c = np.sin(by), np.cos(by)
        axis = Matrices.normalize(v)
        tmp = (1-c)*axis

        rot = np.eye(4)
        rot[0, 0] = c + tmp[0]*axis[0]
        rot[1, 0] = tmp[0]*axis[1] + s*axis[2]
        rot[2, 0] = tmp[0]*axis[2] - s*axis[1]

        rot[0, 1] = tmp[1]*axis[0] - s*axis[2]
        rot[1, 1] = c + tmp[1]*axis[1]
        rot[2, 1] = tmp[1]*axis[2] + s*axis[0]

        rot[0, 2] = tmp[2]*axis[0] + s*axis[1]
        rot[1, 2] = tmp[2]*axis[1] - s*axis[0]
        rot[2, 2] = c + tmp[2]*axis[2]

        return rot

    @staticmethod
    def scale(by):
        return np.diagflat([by, 1])

    @staticmethod
    def normalize(v):
        m = np.linalg.norm(v)
        if m == 0:
            return v
        else:
            return v / m

    @staticmethod
    def look_at(eye, center, up):
        f = Matrices.normalize(center - eye)
        s = Matrices.normalize(np.cross(f, up))
        u = np.cross(Matrices.normalize(s), f)

        lookat = np.eye(4)
        # lookat[0, 0] = s[0]
        # lookat[0, 1] = s[1]
        # lookat[0, 2] = s[2]

        # lookat[1, 0] = u[0]
        # lookat[1, 1] = u[1]
        # lookat[1, 2] = u[2]
        #
        # lookat[2, 0] = -f[0]
        # lookat[2, 1] = -f[1]
        # lookat[2, 2] = -f[2]

        lookat[0, :3] = s
        lookat[1, :3] = u
        lookat[2, :3] = -f

        # lookat[0, 3] = -s.dot(eye)
        # lookat[1, 3] = -u.dot(eye)
        # lookat[2, 3] = f.dot(eye)

        return lookat.dot(Matrices.translate(-eye))

    @staticmethod
    def perspective(fovy, aspect, z_near, z_far):
        tan_half_fovy = np.tan(fovy * np. pi / 180 / 2)
        result = np.zeros((4, 4))
        result[0, 0] = 1 / aspect / tan_half_fovy
        result[1, 1] = 1 / tan_half_fovy
        result[2, 2] = (z_near + z_far) / (z_near - z_far)
        result[3, 2] = -1
        result[2, 3] = 2 * z_far * z_near / (z_near - z_far)

        return result


class Mesh:
    def __init__(self):
        self.vao = -1
        self.vbo = []
        self.vertices = np.empty(1)
        self.indices = np.empty(1)
        self.data_count = 0

    def set_buffers(self):
        color_as_attribute = hasattr(self, 'colors')

        if self.vao == -1:
            self.vao = glGenVertexArrays(1)
            self.vbo = glGenBuffers(3 if color_as_attribute else 2)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo[0])
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)

        # glEnable(GL_ELEMENT_ARRAY_BUFFER)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo[1])
        # glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.indices)*4, (ctypes.c_uint * len(self.indices))(*self.indices), GL_STATIC_DRAW)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)

        if color_as_attribute:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo[2])
            glBufferData(GL_ARRAY_BUFFER, self.colors, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)


class Box(Mesh):
    def __init__(self, half_sizes, colors):
        super().__init__()
        self.vertices = np.array([
            # Bottom
            half_sizes[0], -half_sizes[1], -half_sizes[2],
            half_sizes[0],  half_sizes[1], -half_sizes[2],
            -half_sizes[0],  half_sizes[1], -half_sizes[2],
            -half_sizes[0], -half_sizes[1], -half_sizes[2],
            # Top
            half_sizes[0], -half_sizes[1],  half_sizes[2],
            half_sizes[0], half_sizes[1],  half_sizes[2],
            -half_sizes[0], half_sizes[1],  half_sizes[2],
            -half_sizes[0], -half_sizes[1],  half_sizes[2]
        ])
        self.indices = np.array([
            # Bottom
            0, 1, 2, 2, 3, 0,
            # Front
            0, 1, 5, 5, 4, 0,
            # Right
            1, 2, 6, 6, 5, 1,
            # Back
            2, 3, 7, 7, 6, 2,
            # Left
            3, 0, 4, 4, 7, 3,
            # Top
            4, 5, 6, 6, 7, 4
        ])
        # self.colors = np.array(colors)


class Square(Mesh):
    def __init__(self, sizes):
        super().__init__()
        self.vertices = np.array([
            sizes[0], sizes[1], 0,
            sizes[0], -sizes[1], 0,
            -sizes[0], -sizes[1], 0,
            -sizes[0], sizes[1], 0
        ])
        self.indices = np.array([
            0, 1, 2, 2, 3, 0
        ])


def init():
    global shader_uniform_color
    global shader_attribute_color

    vertex_uniform_color = """
        uniform mat4 mvp;
    
        attribute vec3 pos;
        
        void main()
        {
            //gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0); //gl_Vertex;
            gl_Position = mvp * vec4(pos, 1.0);
        }
    """

    fragment_uniform_color = """
        uniform vec4 color;
    
        void main()
        {
            gl_FragColor = color;
        }
    """

    shader_uniform_color = compileProgram(
        compileShader(vertex_uniform_color, GL_VERTEX_SHADER),
        compileShader(fragment_uniform_color, GL_FRAGMENT_SHADER)
    )

    vertex_attribute_color = """
            uniform mat4 mvp;

            attribute vec3 pos;
            attribute vec4 v_color;
            
            varying vec4 f_color;
            
            void main()
            {
                //gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0); //gl_Vertex;
                gl_Position = mvp * vec4(pos, 1.0);
                f_color = v_color;
            }
        """

    fragment_attribute_color = """
            varying vec4 f_color;

            void main()
            {
                gl_FragColor = f_color;
            }
        """

    shader_attribute_color = compileProgram(
        compileShader(vertex_attribute_color, GL_VERTEX_SHADER),
        compileShader(fragment_attribute_color, GL_FRAGMENT_SHADER)
    )

    global meshes
    meshes = []
    meshes.append(Box([1., 1., 1.],
                      [1.0, 0.0, 0.0, 1.0,
                       0.0, 1.0, 0.0, 1.0,
                       0.0, 0.0, 1.0, 1.0,
                       1.0, 1.0, 0.0, 1.0,
                       1.0, 0.0, 1.0, 1.0,
                       0.0, 1.0, 1.0, 1.0]))

    meshes.append(Square([0.5, 0.4]))

    for mesh in meshes:
        mesh.set_buffers()

    global color
    color = np.array([1.0, 0.0, 0.2, 1.0])

    global loc_mvp_mat_ac, loc_mvp_mat_uc
    loc_mvp_mat_ac = glGetUniformLocation(shader_attribute_color, 'mvp')
    loc_mvp_mat_uc = glGetUniformLocation(shader_uniform_color, 'mvp')
    global loc_color
    loc_color = glGetUniformLocation(shader_uniform_color, 'color')


def reshape(w, h):
    global window_height, window_width
    window_height, window_width = h, w
    glViewport(0, 0, w, h)

    global aspect
    aspect = w / h
    global p_mat
    p_mat = Matrices.perspective(fov, aspect, z_near, z_far)


def set_matrix_cube():
    global m_mat, mv_mat, mvp_mat, v_mat, p_mat
    m_mat = Matrices.translate([0.0, 0.0, 0.0])
    # m_mat = Matrices.rotate(20, 1) * m_mat
    mv_mat = v_mat.dot(m_mat)
    mvp_mat = p_mat.dot(mv_mat)


def set_matrix_square():
    global m_mat, mv_mat, mvp_mat
    m_mat = Matrices.rotate(np.pi/4, np.array([0, 1, 0]))
    mv_mat = v_mat.dot(m_mat)
    mvp_mat = p_mat.dot(mv_mat)


def set_matrix_cone():
    global m_mat, mv_mat, mvp_mat, v_mat, p_mat
    m_mat = Matrices.translate([0.0, 1.0, 0.0])
    mv_mat = v_mat.dot(m_mat)
    mvp_mat = p_mat.dot(mv_mat)


def set_uniforms(color_is_attribute):
    global mvp_mat
    if color_is_attribute:
        global loc_mvp_mat_ac
        glUniformMatrix4fv(loc_mvp_mat_ac, 1, False, mvp_mat)
    else:
        global loc_mvp_mat_uc, loc_color, color
        glUniformMatrix4fv(loc_mvp_mat_uc, 1, False, mvp_mat)
        glUniform4fv(loc_color, 1, color)


def display():
    glEnable(GL_DEPTH_TEST)
    glViewport(0, 0, window_width, window_height)
    # glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
    glClearColor(0.7, 0.7, 0.7, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    global v_mat
    v_mat = Matrices.look_at(eye, eye + target, up)

    glUseProgram(shader_uniform_color)

    set_matrix_square()
    set_uniforms(color_is_attribute=False)
    glutSolidTorus(0.5, 1., 30, 20)

    # glUseProgram(shader_uniform_color)
    #
    # set_matrix_cube()
    # set_uniforms(color_is_attribute=False)
    # meshes[0].draw()

    glUseProgram(0)

    glutSwapBuffers()


def idle():
    if "prev_t" not in idle.__dict__:
        idle.prev_t = 0

    t = glutGet(GLUT_ELAPSED_TIME)
    dt = (t - idle.prev_t) * 1e-3
    idle.prev_t = t

    forward_dir = np.array([-np.sin(angles[0]), 0, np.cos(angles[0])])
    right_dir = np.array([-forward_dir[2], 0, forward_dir[0]])

    global eye
    if move & left:
        eye -= right_dir * move_speed * dt
    if move & right:
        eye += right_dir * move_speed * dt
    if move & forward:
        eye += forward_dir * move_speed * dt
    if move & backward:
        eye -= forward_dir * move_speed * dt
    if move & upward:
        eye[1] += move_speed * dt
    if move & downward:
        eye[1] -= move_speed * dt

    eye = eye.clip(-MAX_EYE_DIST, MAX_EYE_DIST)

    glutPostRedisplay()


def keyboard(i_key, x, y):
    key = i_key.decode('utf-8')

    if key == 'q':
        exit(0)

    global move
    if key == 'w':
        move |= forward
    if key == 'a':
        move |= left
    if key == 's':
        move |= backward
    if key == 'd':
        move |= right
    if key == 'z':
        move |= upward
    if key == 'x':
        move |= downward


def keyboard_up(i_key, x, y):
    key = i_key.decode('utf-8')

    global move
    if key == 'w':
        move &= ~forward
    if key == 'a':
        move &= ~left
    if key == 's':
        move &= ~backward
    if key == 'd':
        move &= ~right
    if key == 'z':
        move &= ~upward
    if key == 'x':
        move &= ~downward


def motion(pos_x, pos_y):
    if "wrap" not in motion.__dict__:
        motion.wrap = False

    if not motion.wrap:
        window_size = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)

        global angles
        angles += mouse_speed * (np.array([pos_x, pos_y]) - np.array(window_size) / 2)

        if angles[0] < - np.pi:
            angles[0] += 2 * np.pi
        elif angles[0] > np.pi:
            angles[0] -= 2 * np.pi

        if angles[1] < -np.pi / 2 + 1e-3:
            angles[1] = -np.pi / 2 + 1e-3
        elif angles[1] > np.pi / 2 - 1e-3:
            angles[1] = np.pi / 2 - 1e-3

        global target
        target = np.array([
            -np.sin(angles[0]) * np.cos(angles[1]),
            -np.sin(angles[1]),
            np.cos(angles[0]) * np.cos(angles[1])
        ])
        motion.wrap = True
        glutWarpPointer(window_size[0] // 2, window_size[1] // 2)
    else:
        motion.wrap = False


def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    return shader


def launch():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    glutInitWindowPosition(50, 50)
    glutCreateWindow(b"Hello")

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)
    glutKeyboardFunc(keyboard)
    glutKeyboardUpFunc(keyboard_up)
    glutMotionFunc(motion)
    glutPassiveMotionFunc(motion)
    glutSetCursor(GLUT_CURSOR_NONE)

    try:
        init()
    except BaseException as e:
        print('Failed init: ' + str(e))
        exit(-1)

    try:
        glutMainLoop()
    except BaseException as e:
        print('Failed main loop: ' + str(e))


launch()
