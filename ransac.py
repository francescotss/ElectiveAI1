import numpy as np

class RansacModel:

    
    def __init__(self, sample_percentage=0.25, outliers_probability=0.5, success_probability=0.95):
        self._s = sample_percentage 
        self._eps = outliers_probability
        self._p = success_probability
    
    @property
    def eps(self):
        return self._eps
    
    @property
    def p(self):
        return self._p
    
    
    @property
    def s(self):
        return self._s
    
    def iteration_number(self):
        # N =  log(1-p) / (log(1-(1-eps))^s)
        # s = number of sample
        # eps = percentage of outliers in the data
        # p = requested probability of success
        N = np.log(1-self.p) / np.log(1-(1-self.eps)^self.s)
        return N
    
    def run(self, data):
        iteration_done = 0
        num_iteration = self.iteration_number
        
        data_size = data.shape
        
        sample_num = int(data_size*self.s)
        
        
        
        # _error = compute_error()
        while num_iteration > iteration_done:
            # error = compute_error()
            if error < _error:
                # new solution
                
        
    
    def fit(self, A, Y, num_sample, threshold):

        num_iterations = math.inf
        num_sample = 3

        max_inlier_count = 0
        best_model = None

        prob_outlier = 0.5
        desired_prob = 0.95

        total_data = np.column_stack((A, Y))  ## [ A | Y]
        data_size = len(total_data)

        iterations_done = 0
        # Adaptively determining the number of iterations
        while num_iterations > iterations_done:

            # shuffle the rows and take the first 'num_sample' rows as sample data
            np.random.shuffle(total_data)
            sample_data = total_data[:num_sample, :]
            
            estimated_model = self.curve_fitting_model.fit(sample_data[:,:-1], sample_data[:, -1:]) ## [a b c]

            # count the inliers within the threshold
            y_cap = A.dot(estimated_model)
            err = np.abs(Y - y_cap.T)
            inlier_count = np.count_nonzero(err < threshold)

            # check for the best model 
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_model


            prob_outlier = 1 - inlier_count/data_size
            print('# inliers:', inlier_count)
            print('# prob_outlier:', prob_outlier)
            num_iterations = math.log(1 - desired_prob)/math.log(1 - (1 - prob_outlier)**num_sample)
            iterations_done = iterations_done + 1

            print('# s:', iterations_done)
            print('# n:', num_iterations)
            print('# max_inlier_count: ', max_inlier_count)

        return best_model
