class ForwardUpdateEquation(object):

    @property
    def num_control_params(self):
        """
        Number of control parameters expected.
        Returns: int
        """
        raise NotImplementedError()

    def neg_elbo(self, *args):
        """
        Return the negative ELBO.
        Args:
            *args:

        Returns:

        """
        raise NotImplementedError()

    def forward_model(self, mu, *control_params):
        """
        Return the model data.
        Args:
            mu: [K]
            *control_params: list of any other arrays
        Returns:
            Model data [N]

        """
        raise NotImplementedError()

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        """
        Given the current data and control params as well as a Gaussian prior, return the conditional mean and covariance
        of a Gaussian variational posterior.

        Args:
            prior_mu: [K] prior mean
            prior_Gamma: [K,K] prior covariance
            Y: [N] observed data
            Sigma: [N,N] Observed data covariance
            *control_params: list of arrays of arbitrary shape.

        Returns:
            posterior mean [K]
            posterior covariance [K,K]
        """
        return prior_mu, prior_Gamma


