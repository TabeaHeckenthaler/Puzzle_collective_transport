from Analysis.grid import *
from scipy.stats import wasserstein_distance


# I want to load the CDFs of the ant experiments and gillespie
class Compare:
    def __init__(self, grid1, grid2):
        self.grid1 = grid1
        self.grid2 = grid2

    def save_grid_distances(self):
        dist = {shape: {size_exp: {} for size_exp in ['XL', 'M']} for shape in shapes}
        for size_exp, shape in grid_exp:
            for cal_sim in grid_sim.__iter__(size_exp + ' ' + shape):
                if cal_sim.shape != shape:
                    continue
                hit_stats_sim, hit_stats_exp = comp.get_hit_stats(cal_sim, shape, size_exp)
                dist[shape][size_exp][str(cal_sim)] = comp.get_distance_CDFs(hit_stats_sim, hit_stats_exp, size_exp, shape, 'pL')

        # save dist in json file
        with open('distance_CDFs.json', 'w') as f:
            json.dump(dist, f)

    def find_smallest_distance_for_each_shape_and_size(self):
        with open('distance_CDFs.json', 'r') as f:
            dist = json.load(f)

        min_dist = {shape: {} for shape in shapes}

        for shape in shapes:
            for size_exp in ['M', 'XL']:
                min_dist[shape][size_exp] = min(dist[shape][size_exp].items(), key=lambda x: x[1])[0]

        with open('min_distance_CDFs.json', 'w') as f:
            json.dump(min_dist, f)

    def get_hit_stats(self, cal_sim, shape, size_exp):
        df_ant = get_exp_df()
        df_ant = df_ant[df_ant['size'] == size_exp]
        df_ant = df_ant[df_ant['shape'] == shape]
        hit_stats_exp = HIT_statistics(df_ant.copy(),
                                       results_folder=directory_results + '\\HIT_stats\\',
                                       solver_string='ant_HIT')

        hit_stats_sim = HIT_statistics(cal_sim.get_df(),
                                       results_folder=cal_sim.folder(),
                                       solver_string='gillespie_HIT')

        return hit_stats_sim, hit_stats_exp

    @staticmethod
    def get_distance_CDFs(hit_stats_sim, hit_stats_exp, size_exp, shape, string):
        successful_attempt1, unsuccessful_attempt1, not_attempt1 = hit_stats_sim.load_eff(string=string)
        sorted_lengths1, y_values_solved1, _, _, _ = (
            hit_stats_sim.get_CDF_per_attempt(successful_attempt1, unsuccessful_attempt1, 'L', shape,
                                              maxx_time=10, string=string))

        successful_attempt2, unsuccessful_attempt2, not_attempt2 = hit_stats_exp.load_eff(string=string)
        sorted_lengths2, y_values_solved2, _, _, _ = (
            hit_stats_exp.get_CDF_per_attempt(successful_attempt2, unsuccessful_attempt2, size_exp, shape,
                                           maxx_time=10, string=string))

        # how can I get the wasserstein distance though sorted_lengths1 and sorted_lengths2 are not the same?
        common_x = np.linspace(min(sorted_lengths1[0], sorted_lengths2[0]), max(sorted_lengths1[-1], sorted_lengths2[-1]), num=1000)
        y_values_solved1_interp = np.interp(common_x, sorted_lengths1, y_values_solved1)
        y_values_solved2_interp = np.interp(common_x, sorted_lengths2, y_values_solved2)

        # calculate the wasserstein distance
        wasserstein_dist = wasserstein_distance(y_values_solved1_interp, y_values_solved2_interp)
        return wasserstein_dist

class GridIterator_exp:
    def __init__(self):
        self.sizes = ['M', 'XL']
        self.shapes = shapes

    def __iter__(self):
        for size in self.sizes:
            for shape in self.shapes:
                yield size, shape

if __name__ == '__main__':

    gravnon_uniform_force_corner_rnd_corner = [(True, True, False)]
    keys = ['_'.join(map(str, item)) for item in gravnon_uniform_force_corner_rnd_corner]
    prob_rnds = [0.05, 0.2, 0.4]  #1
    kappas = [1, 5, 8] # , 32
    parameters = [(p, k, None) for (p, k) in product(prob_rnds, kappas)]
    grid_sim = GridIterator({k: parameters for k in keys}, shapes)

    grid_exp = GridIterator_exp()
    #
    comp = Compare(grid_sim, grid_exp)
    comp.save_grid_distances()
    comp.find_smallest_distance_for_each_shape_and_size()
    # DEBUG = 1

    same_parameters = False
    grid = {}
    for string in ['pL', 'time']:
        best_grid = GridIterator.best_grid()
        best_grid.plot_CDF(string=string, max_time=30)
        plt.savefig(os.path.join(directory_results, f'CMD_specific_simulation_{string}.pdf'))
