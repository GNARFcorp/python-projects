import random, math, time
import itertools, hashlib
import numpy as np


start_time = time.time()


def print_time(message):
    global start_time
    print(str(time.time() - start_time).ljust(20), message)


class ChunkManager(object):
    def __init__(self, n_dim=2, octaves=4, persistance=0.5, tile_len=64,
                 seed=time.time()):
        self.seed = str(seed).encode("utf-8")
        self.tile_len = tile_len
        self.chunks = {}
        self.n_dim = n_dim
        self.octaves = octaves
        self.persistence = persistance

    def __getitem__(self, *args):
        if len(*args) is not self.n_dim:
            raise KeyError("Key dimensions %s doesn't match the chunk dimension of %s!" %(tuple(*args), self.n_dim))
        coord = tuple(*args)
        if coord not in self.chunks:
            self.get_perlin_chunk(coord)
        return self.chunks[coord]

    def determ_rand_vector(self, salt):
        # Function for providing deterministic random values based on an input seed
        # n_values is optional and sets the numbers of output values

        if type(salt) == list:
            salt = tuple(salt)
        random.seed(hashlib.md5(salt.encode("utf-8") + self.seed).hexdigest())

        if self.n_dim == 1:
            rand_vector = random.choice((-1, 1))
        else:
            found = False
            while found is False:
                rand_vector = []
                for x in range(self.n_dim):
                    rand_vector.append(random.uniform(-1, 1))
                rand_vector = np.array(rand_vector)
                length = math.sqrt(sum(rand_vector ** 2))
                if length < 1:
                    rand_vector = rand_vector / length
                    found = True
                elif length == 1:
                    found = True
        return rand_vector

    def get_gradients(self, coordinates):
        # Gives gradients of the corners of a n_dimensional point
        coordinates = np.array(coordinates)
        gradient_lst = []
        corner_lst = self.ndim_grid(coordinates, coordinates + 2)
        for corner in corner_lst:
            gradient_lst.append(self.determ_rand_vector(str(corner)))
        gradient_lst = np.array(gradient_lst)
        return gradient_lst

    def dot_gradients(self, gradient_lst, coordinates):
        dot_product = []
        dist = np.array(coordinates) % 1
        dist_neg = dist - 1
        dist_lst = list(itertools.product(*zip(dist, dist_neg)))
        for n, dist in enumerate(dist_lst):
            dot_product.append(sum(dist * gradient_lst[n]))
        return dot_product

    def ndim_grid(self, start, stop, step=1):
        # Works like range(start, stop, step), but in n-dimensional space.
        # Set number of dimensions
        ndims = len(start)

        # List of ranges across all dimensions
        L = [np.linspace(start[i], stop[i], (stop[i] - start[i]) * step,
                        endpoint = False) for i in range(ndims)]

        # Finally use meshgrid to form all combinations corresponding to all
        # dimensions and stack them as M x ndims array
        return np.hstack((np.meshgrid(*L))).swapaxes(0, 1).reshape(ndims, -1).T

    def interpol(self, point_a, point_b, dist):
        # interpolates the value of point P, takes int or np-array
        point_a, point_b = np.array(point_a), np.array(point_b)
        factor = 6 * dist ** 5 - 15 * dist ** 4 + 10 * dist ** 3
        point_p = (point_b - point_a) * factor + point_a
        return point_p

    def interpol_corners(self, value_lst, coordinates):
        # Takes the corners of a cube and interpolates them down
        for n_axis, axis in enumerate(reversed(coordinates)):
            dist = axis % 1
            factor = 6 * dist ** 5 - 15 * dist ** 4 + 10 * dist ** 3
            for n in range(0, len(value_lst) // (n_axis + 1) ** 2, 2):
                value_lst[n // 2] = np.array((value_lst[n + 1] - value_lst[n]) * factor + value_lst[n])
        return float(value_lst[0])

    def get_perlin_chunk(self, coord):
        # Generates a n-dimensional cubic field of noise with tile_len values per axis
        coord = np.array(coord)
        coord_map = self.ndim_grid(coord, coord + 1, self.tile_len)
        gradients = self.get_gradients(coord)
        noise_map = list(map(lambda index: self.interpol_corners(
                self.dot_gradients(gradients, index), index), coord_map))
        shape = [self.tile_len] * self.n_dim
        self.chunks[tuple(coord)] = np.array(noise_map).reshape(shape)

    def gen_value_chunk(self, coord):
        for n in range(self.octaves):
            frequency = 2 ** n
            amplitude = persistence ** n

    def perlin_noise_nd(self, lower_corner, upper_corner, tile_len, seed):
        index_map = ndim_grid(lower_corner, upper_corner)
        chunk_map = {}
        for n, index in enumerate(index_map):
            chunk_map[tuple(index)] = get_perlin_chunk(index, tile_len, seed)
        return chunk_map

    def blend_noise(self, lower_corner, upper_corner):
        period_length = 1
        for n in range(nmbr_octaves):
            period_length *= 2
            noise_maps.append(perlin_noise)
        pass
        # TODO: Actually do something here

    def plot_2d(self, start, stop):
        surf.fill((0, 0, 0))
        screen.fill((0,  0, 0))
        for x_chunk in range(start[0], stop[0]):
            for y_chunk in range(start[1], stop[1]):
                coord = np.array((x_chunk, y_chunk))
                print_time("Generating chunk %s now" %coord)
                print_time("Finished generating. Starting conversion.")
                tile = ((self[tuple(coord)] + 1) * 128).astype(int)
                tmp_surf = pygame.surfarray.make_surface(tile)
                surf.blit(pygame.transform.scale(tmp_surf, (128, 128)), coord * 128)
                screen.blit(surf, (0, 0))
                pygame.display.flip()

    def plot_1d(self, axis, start, length, amplitude=10):
        for current_step in range(start[axis], start[axis] + length):
            index = start
            index[axis] = current_step
            chunk = np.swapaxes(self[tuple(index)], axis, 0)
            for value in chunk[0]:
                print(((1 + value) * amplitude) * "#")

    def join_2d_field(self, start, stop):
        field = []
        for x in range(start[0], stop[0]):
            line = np.concatenate((self.chunks[(x, start[1])], self.chunks[(x, start[1]+1)]), axis=1)
            for y in range(start[1]+2, stop[1]):
                line = np.concatenate(line, self.chunks[(x, y)], axis=1)
            field.append(line)
        print(field)
        return np.array(field)

chunks = ChunkManager(seed="hai", tile_len=16, n_dim=5)
#chunks.plot_1d(4,[1,1,0,4,5],30, 80)
#print(chunks.join_2d_field((0, 0), (0, 2)))
#chunk = perlin_noise_nd([3, 0], [10, 10], 16, "hoi")
#print(chunk[(3, 0)], chunk[(4, 0)])
#print(np.concatenate((chunk[(3, 0)], chunk[(4, 0)]), axis=0))
import pygame
pygame.init()

surf = pygame.Surface((1000, 1000))
surf.fill((0, 0, 0))

clock = pygame.time.Clock()

width = 1000
height = 1000

screen = pygame.display.set_mode((width, height))
chunks.tile_len = 4
for x in range(4):
    chunks.plot_2d([0, 0], [5, 5])
    chunks.tile_len *= 2
    print(chunks.tile_len)
    time.sleep(2)


running = True
"""
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    clock.tick(60)
"""
1/0
coordinates = [0, 0]
gradients = get_gradients([3, 6], seed="hoi")
corner = [0, 0]
x_range = map((lambda x: x / 10.0), range(0, 10))
y_range = map((lambda x: x / 10.0), range(0, 10))
for c_x in range(upper_corner[0]):
    print(corner)
    for c_y in range(upper_corner[1]):
        corner = [c_x, c_y]
#        print(corner[0], corner[1], "%", sep="")
        gradients = get_gradients(corner, seed="hoi")
#        print(coordinates)
        for x in x_range:
            coordinates[0] = x + c_x
            for y in y_range:
                coordinates[1] = y + c_y
                values = dot_gradients(gradients, coordinates)
                noise = ((interpol_corners(values, coordinates) + 1) / 2) * 255
                color = [noise] * 3
#                if noise <= 0.35:
#                    noise = (noise - 0.2) * 6.666666
#                    color = interpol((0, 0, 80), (0, 178, 238), noise)
#                elif 0.35 < noise <= 0.4:
#                    color = (255, 215, 0)
#                    noise = (noise - 0.35) * 20
#                    color = interpol((255, 215, 0), (205, 205, 0), noise)
#                elif 0.35 < noise <= 0.5:
#                    noise = (noise - 0.35) * 6.66
#                    color = interpol((128, 128, 0), (173, 255, 47), noise)
#                elif 0.5 < noise <= 0.7:
#                    noise = (noise - 0.5) * 5
#                    color = interpol((180, 200, 50), (200, 200, 200), noise)
#                elif 0.7 < noise:
#                    noise = (noise - 0.7) * 10
#                    color = interpol((230, 230, 230), (255, 255, 255), noise)
#		print((int(round(coordinates[0] * 10)), int(round(coordinates[1] * 10))))
#                print(coordinates)

                try:
                    surf.set_at((int(round(coordinates[0]*10)), int(round(coordinates[1]*10))),
                        color)
                except:
                    print(noise, color, coordinates)

