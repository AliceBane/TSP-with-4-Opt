#!/usr/bin/python3
import math
import time
import math
from TSPClasses import TSPSolution

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

from TSPClasses import *
import heapq as hq
import copy


# Thank you to my classmates for helping me understand this code better and giving me examples on how to
# solve the more complex portions of this problem. I've rewritten and created all my own methods instead of copying
# for my own sake of knowledge and originality (even if it meant a sacrifice in performance). All work should be
# Authentic, even if based upon prior concepts.

class TSPSolver:
	def __init__(self, gui_view):
		self._scenario = None

	def setup_with_scenario(self, scenario):
		self._scenario = scenario

	def find_closest_city(self, city, list_of_cities: list):
		min_dist = math.inf
		min_city = None
		for city in list_of_cities:
			dist = city.costTo(city)
			if dist < min_dist:
				min_city = city
				min_dist = dist
		return min_city

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		n_cities = len(cities)
		found_tour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not found_tour and time.time() - start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation(n_cities)
			route = []
			# Now build the route using the random permutation
			for i in range(n_cities):
				route.append(cities[perm[i]])
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				found_tour = True
		end_time = time.time()
		results['cost'] = bssf.cost if found_tour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	# For each starting city, the function tries to find the shortest path that visits all
	# remaining cities using the greedy approach.
	def greedy(self, time_allowance=60.0):
		cities = self._scenario.getCities()
		start_time = time.time()
		best_solution = None
		num_solutions = 0

		# Loop through each city as a starting point
		for start_city in cities:
			if time.time() - start_time > time_allowance:
				break

			route = [start_city]
			remaining_cities = set(cities)
			remaining_cities.remove(start_city)

			# Add nearest city to the current city until all cities have been visited
			while remaining_cities:
				next_city = min(remaining_cities, key=lambda city: start_city.costTo(city))
				if start_city.costTo(next_city) == math.inf:
					break

				route.append(next_city)
				remaining_cities.remove(next_city)
				start_city = next_city

			# If valid solution is found, update variables for best solution and number of solutions
			if len(route) == len(cities) and start_city.costTo(route[0]) != math.inf:
				solution = TSPSolution(route)
				num_solutions += 1

				if not best_solution or solution.cost < best_solution.cost:
					best_solution = solution

		end_time = time.time()
		results = {
			'cost': best_solution.cost if best_solution else math.inf,
			'time': end_time - start_time,
			'count': num_solutions,
			'soln': best_solution,
			'max': None,
			'total': None,
			'pruned': None,
		}
		return results

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	def branchAndBound(self, time_allowance=60.0):
		results = {}
		count, total, maximum, pruned = 0, 0, 0, 0
		greedy_result = self.greedy()
		time_start = time.time()

		state_heap = []

		# get greedy solution and set it as the current best solution
		greedy = greedy_result['soln']
		best_so_far = TSPSolution(greedy.getRoute())
		route_length = len(greedy.getRoute())
		start_state = StateBranchAndBound(greedy.getRoute())
		hq.heappush(state_heap, start_state)
		total += 1

		# loop until there are no more states to explore or time limit is reached
		while len(state_heap) > 0:
			if time.time() - time_start > time_allowance:
				break
			maximum = max(maximum, len(state_heap))

			state = hq.heappop(state_heap)
			if len(state.currentPath) == route_length:
				solution = TSPSolution(state.currentPath)
				if best_so_far.cost > solution.cost:
					best_so_far = solution
					count += 1
				else:
					pruned += 1

			elif (state.lowerBound < math.inf) and (state.lowerBound < best_so_far.cost):
				count += 1
				new_states = state.possible_state()
				for newState in new_states:
					hq.heappush(state_heap, newState)
					total += 1
			else:
				pruned += 1

		end_time = time.time()
		results.update({
			'cost': best_so_far.cost,
			'time': greedy_result['time'] + end_time - time_start,
			'count': count,
			'soln': best_so_far,
			'max': maximum,
			'total': total,
			'pruned': pruned
		})
		return results

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

	# Implementation uses a random restart check, this can prevent it from getting stuck in the local minimum.
	# While this can (and usually will) produce a better solution than that of greedy, it more than likely won't
	# find the optimal solution to the problem. So for the trade of a small amount of time (which can made shorter
	# if num_restarts is lowered) you can find an improved path than that of greedy or randomly creating a path.
	def fancy(self, time_allowance=60.0, num_restarts=15):
		'''results = {}
		time_start = time.time()

		num_solutions = 0
		greedy_result = self.greedy()
		greedy = greedy_result['soln']
		best_so_far = TSPSolution(greedy.getRoute())
		best_distance = best_so_far.cost

		for restart in range(num_restarts):
			if time.time() - time_start > time_allowance:
				break

			if restart > 0:
				random.shuffle(greedy.getRoute()[1:-1])  # Shuffle all nodes except first and last
				current_solution = TSPSolution(greedy.getRoute())
			else:
				current_solution = best_so_far

			improved = True
			while improved:
				if time.time() - time_start > time_allowance:
					break

				improved = False
				for i in range(1, len(current_solution.getRoute()) - 1):
					for j in range(i + 1, len(current_solution.getRoute())):
						if time.time() - time_start > time_allowance:
							break

						new_route = current_solution.getRoute()[:]
						new_route[i:j] = current_solution.getRoute()[i:j][::-1]
						new_distance = TSPSolution(new_route).cost
						if new_distance < best_distance:
							num_solutions += 1
							best_distance = new_distance
							best_route = new_route
							improved = True

				if improved:
					current_solution = TSPSolution(best_route)

			if current_solution.cost < best_so_far.cost:
				best_so_far = current_solution

		end_time = time.time()
		results = {
			'cost': best_so_far.cost if best_so_far else math.inf,
			'time': end_time - time_start,
			'count': num_solutions,
			'soln': best_so_far,
			'max': None,
			'total': None,
			'pruned': None,
		}
		return results

		VERSION 2
		results = {}
		time_start = time.time()

		num_solutions = 0
		greedy_result = self.greedy()
		greedy = greedy_result['soln']
		best_so_far = TSPSolution(greedy.getRoute())
		best_distance = best_so_far.cost

		# Define dynamic time allowance parameters
		initial_time_allowance = time_allowance
		time_allowance_decrement = 0.02
		min_time_allowance = 1.0

		for restart in range(num_restarts):
			if time.time() - time_start > time_allowance:
				break

			if restart > 0:
				random.shuffle(greedy.getRoute()[1:-1])
				current_solution = TSPSolution(greedy.getRoute())
			else:
				current_solution = best_so_far

			# Dynamic time allowance adjustment
			time_allowance = initial_time_allowance - time_allowance_decrement * restart
			time_allowance = max(time_allowance, min_time_allowance)

			# Perform 2-Opt local search
			improved = True
			while improved and time.time() - time_start < time_allowance:
				improved = False
				for i in range(1, len(current_solution.getRoute()) - 2):
					for j in range(i + 1, len(current_solution.getRoute()) - 1):
						new_route = self._two_opt_swap(current_solution.getRoute(), i, j)
						new_distance = TSPSolution(new_route).cost
						if new_distance < best_distance:
							num_solutions += 1
							best_distance = new_distance
							best_route = new_route
							improved = True

				if improved:
					current_solution = TSPSolution(best_route)

			if current_solution.cost < best_so_far.cost:
				best_so_far = current_solution

		end_time = time.time()
		results = {
			'cost': best_so_far.cost if best_so_far else math.inf,
			'time': end_time - time_start,
			'count': num_solutions,
			'soln': best_so_far,
			'max': None,
			'total': None,
			'pruned': None,
		}
		return results

	# Helper function for 2-Opt local search
	def _two_opt_swap(self, route, i, j):
		new_route = route[:i] + route[i:j][::-1] + route[j:]
		return new_route '''

	def fancy(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()

		# Precompute distances
		dist_matrix = np.zeros((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				dist_matrix[i, j] = cities[i].costTo(cities[j])

		while not foundTour and time.time() - start_time < time_allowance:
			route = []
			visited = set()
			current_city_index = np.random.randint(ncities)  # Start from a random city
			route.append(current_city_index)
			visited.add(current_city_index)

			while len(route) < ncities:
				min_cost = np.inf
				next_city_index = None
				for city_index in range(ncities):
					if city_index not in visited:
						cost = dist_matrix[current_city_index, city_index]
						if cost < min_cost:
							min_cost = cost
							next_city_index = city_index
				if next_city_index is not None:
					route.append(next_city_index)
					visited.add(next_city_index)
					current_city_index = next_city_index

				if time.time() - start_time > time_allowance:
					break
			route_cities = [cities[i] for i in route]
			bssf = TSPSolution(route_cities)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

class StateBranchAndBound:
	def __init__(self, cities_list, lower_bound=0, current_path=None, prev_state=None, test=None):
		if current_path is None:
			current_path = []
		self.listCities = cities_list
		self.lowerBound = lower_bound
		self.currentPath = copy.copy(current_path)
		self.prevState = prev_state
		self.testPath = test
		self.matrix = Table(len(cities_list))
		self.currentPath.append(cities_list[test[1]] if test else cities_list[0])
		for row in range(len(cities_list)):
			for col in range(len(cities_list)):
				if test and (row == test[1] or col == test[0]):
					self.matrix.set_value(row, col, 'x')
				elif test and row == test[0] and col == test[1]:
					self.matrix.set_value(row, col, np.inf)
				elif not test or col != test[0] or row != test[1]:
					matrix_value = cities_list[col].costTo(cities_list[row]) if not prev_state else prev_state.get_value(
						row,
						col)
					self.matrix.set_value(row, col, matrix_value)
		self.lowerBound += self.matrix_reduce()

	def possible_state(self):
		current_path = self.currentPath
		test = self.testPath

		city_list = self.listCities
		state_matrix = self.matrix
		state_list = []

		lower_bound = self.lowerBound
		current_index = 0

		if test is not None:
			current_index = test[1]

		# Looks a tad bit weird but performance gain from doing it this way is very noticeable (0.02-0.1 second speed up)
		state_list += [
			StateBranchAndBound(city_list, lower_bound + state_matrix.get_value(nextIndex, current_index), current_path,
								state_matrix, [current_index, nextIndex])
			for nextIndex in range(len(city_list))
			if city_list[nextIndex] not in current_path and nextIndex != current_index
		]
		return state_list

	def __lt__(self, other):
		return self.lowerBound < other.lowerBound

	def matrix_reduce(self):
		size_matrix = self.matrix.get_size()
		state_matrix = self.matrix
		lower_bound = 0

		# Reduce rows
		for y in range(size_matrix[1]):
			row_min = np.inf
			for x in range(size_matrix[0]):
				test_value = state_matrix.get_value(x, y)
				if test_value == 'x':
					continue
				elif test_value < row_min:
					row_min = test_value
			if row_min != np.inf:
				lower_bound += row_min
				for x in range(size_matrix[0]):
					test_value = state_matrix.get_value(x, y)
					if test_value != 'x':
						state_matrix.set_value(x, y, test_value - row_min)

		# Reduce columns
		for x in range(size_matrix[0]):
			col_min = np.inf
			for y in range(size_matrix[1]):
				test_value = state_matrix.get_value(x, y)
				if test_value == 'x':
					continue
				elif test_value < col_min:
					col_min = test_value

			if col_min != np.inf:
				lower_bound += col_min
				for y in range(size_matrix[1]):
					test_value = state_matrix.get_value(x, y)
					if test_value != 'x':
						state_matrix.set_value(x, y, test_value - col_min)

		return lower_bound
