import numpy as np
import torch


class Simcim:
    """
    SimCIM (Simulated Coherent Ising Machine) implementation for solving QUBO problems.
    """
    params_disc = {}
    params_cont = {}

    # Continuous parameters for system dynamics
    params_cont['c_th'] = 1.  # Threshold for amplitudes
    params_cont['zeta'] = 1.  # Coupling strength
    params_cont['dt'] = 0.09  # Time step size
    params_cont['sigma'] = 0.1  # Standard deviation for noise
    params_cont['alpha'] = 0.09  # Momentum coefficient

    # Pump parameters
    params_cont['S'] = torch.tensor(0.2)
    params_cont['D'] = torch.tensor(-0.1)
    params_cont['O'] = torch.tensor(0.05)

    # Discrete parameters for system execution
    params_disc['N'] = 300  # Number of time iterations
    params_disc['attempt_num'] = 700  # Number of runs


    def __init__(self, J, b, device, datatype):
        """
         Initialize the SimCIM solver with given interaction matrix and bias vector.

         Args:
             J (torch.Tensor): Interaction matrix.
             b (torch.Tensor): Bias vector.
             device (str): Computation device ('cpu' or 'cuda').
             datatype (torch dtype): Data type for tensor computations.
         """
        self.J = J.type(datatype).to(device)
        self.b = b.type(datatype).to(device)
        self.Jmax = torch.max(torch.sum(torch.abs(J), 1))
        self.dim = J.shape[0]

        self.device = device
        self.datatype = datatype

    def ampl_inc(self, c, p):
        """
        Compute amplitude increment based on system dynamics.

        Args:
            c (torch.Tensor): Current amplitudes.
            p (torch.Tensor): Pumping values.

        Returns:
            torch.Tensor: Updated amplitude values.
        """
        return ((p * c + self.zeta * (torch.mm(self.J, c) + self.b)) * self.dt
                + (self.sigma * torch.randn((c.size(0), self.attempt_num), dtype=self.datatype).to(self.device)))

    def pump(self):
        """
        Generate the pump values over iterations based on hyperparameters.

        Returns:
            torch.Tensor: Pump values array.
        """
        i = torch.arange(self.N, dtype=self.datatype).to(self.device)
        arg = torch.tensor(self.S, dtype=self.datatype).to(self.device) * (i / self.N - 0.5)
        return self.Jmax * self.O * (torch.tanh(arg) + self.D)

    def pump_lin(self):
        t = self.dt * torch.arange(self.N, dtype=self.datatype).to(self.device)
        eigs = torch.eig(self.J)[0][:, 0]
        eig_min = torch.min(eigs)
        eig_max = torch.max(eigs)
        p = -self.zeta * eig_max + self.zeta * (eig_max - eig_min) / t[-1] * t
        return p

    # Amplitude initializer
    def init_ampl(self):
        """
        Initialize the amplitudes tensor with zeros.

        Returns:
            torch.Tensor: Initialized amplitude values.
        """
        return torch.zeros((self.dim, self.attempt_num), dtype=self.datatype).to(self.device)

    def tanh(self, c):
        return self.c_th * torch.tanh(c)

    # Evolution of Amplitudes
    # N -- number of time iterations
    # attempt_num -- number of runs
    # J -- coupling matrix
    # b -- biases
    # O, S, D -- pump parameters
    # sigma -- sqrt(std) for random number
    # alpha -- momentum parameter
    # c_th -- restriction on amplitudes growth
    def evolve(self):
        """
        Perform amplitude evolution for QUBO minimization.

        Returns:
            tuple: (final amplitudes, evolution history, convergence step).
        """
        self.N = self.params_disc['N']
        self.attempt_num = self.params_disc['attempt_num']
        self.dt = self.params_cont['dt']
        self.zeta = self.params_cont['zeta']
        self.c_th = self.params_cont['c_th']
        self.O = self.params_cont['O']
        self.D = self.params_cont['D']
        self.S = self.params_cont['S']
        self.sigma = self.params_cont['sigma']
        self.alpha = self.params_cont['alpha']
        self.Jmax = torch.max(torch.sum(torch.abs(self.J), 1))
        self.dim = self.J.shape[0]

        # Threshold for convergence
        convergence_threshold = -0.1  # set an appropriate threshold
        convergence_step = self.N  # to record the step at which convergence happens
        is_converged = False
        converge_counter = 0

        # Choosing an attempt for the amplitude evolution
        random_attempt = np.random.randint(self.attempt_num)

        # Initialize amplitudes matrix for all variables and attempts
        c_current = self.init_ampl()

        # Initializing full array of amplitudes
        #     c_full = torch.zeros(N,dim,attempt_num)
        #     c_full[0] = c_current

        # Creating the array for evolving amplitudes from random attempt
        c_evol = torch.empty((self.dim, self.N), dtype=self.datatype).to(self.device)
        c_evol[:, 0] = c_current[:, random_attempt]

        # Define pump array
        p = self.pump()
        #     p = self.pump_lin()

        # Define coupling growth
        #     zeta = coupling(init_value,final_value,dt,N)

        # Initializing moving average of amplitudes increment
        dc_momentum = torch.zeros((self.dim, self.attempt_num), dtype=self.datatype).to(self.device)
        #     free_energy_ar = torch.empty(self.N-1, dtype = self.datatype).to(device)

        for i in range(1, self.N):
            c_prev = c_current  # Store previous amplitude state
            # Calculating amplitude increment
            dc = self.ampl_inc(c_current, p[i])
            dc /= torch.sqrt((dc ** 2).sum(0)).reshape(1, self.attempt_num)

            # Calculating moving average of amplitudes increment
            dc_momentum = self.alpha * dc_momentum + (1 - self.alpha) * dc

            # Calculating possible values of amplitudes on the next step
            c1 = c_current + dc_momentum

            # Comparing c1 with c_th
            th_test = (torch.abs(c1) < self.c_th).type(self.datatype)

            # Updating c_current
            #         c_current = c_current + th_test*dc_momentum
            c_current = th_test * (c_current + dc_momentum) + (1. - th_test) * torch.sign(c_current) * self.c_th
            #         c_current = step(c_current + dc_momentum,c_th,device, datatype)
            #         c_current = tanh(c1,c_th)

            # Updating c_full
            #         c_full[i] = torch.tanh(c_full[i-1] + dc_momentum)

            # Check for convergence (if it hasn't already been reached)
            if not is_converged:
                amplitude_change = torch.sum(c_prev) - torch.sum(c_current)
                if amplitude_change > convergence_threshold: # and amplitude_change < 0
                    if converge_counter == 0:
                        convergence_step = i
                    converge_counter += 1
                    if converge_counter > 100:
                        is_converged = True

                else:
                    converge_counter = 0
                    convergence_step = i

            if is_converged:
                break


            # Add amplitude values from random attempt to c_evol array
            c_evol[:, i] = c_current[:, random_attempt]

            # if is_converged:
            #     break

            # s = torch.sign(c_current)
            # free_energy_ar[i-1] = self.free_energy(s,self.beta)

        return c_current, c_evol, convergence_step  # , free_energy_ar

    def energy(self, s):
        """
        Compute the energy of a given solution based on the QUBO formulation.

        Args:
            s (torch.Tensor): Binary solution vector.

        Returns:
            torch.Tensor: Computed energy.
        """
        return -0.5 * torch.einsum('in,ij,jn->n', s, self.J, s) - torch.einsum('in,ik->n', s, self.b)
