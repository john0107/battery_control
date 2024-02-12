from sklearn.preprocessing import PolynomialFeatures


class BaseFeatureTransformer():
    def __init__(self, sim, states=('price', ), all_states=False):
        self.states = states
        for state in self.states:
            assert state in sim.get_state_labels(), (state, sim.get_state_labels())
        if all_states:  # all states
            self.idx = list(range(len(sim.get_state_labels())))
        else:  # only the states specified
            self.idx = [i for i, state in enumerate(
                sim.get_state_labels()) if state in self.states]

    def transform(self, X):
        raise NotImplementedError


class PolynomialTransformer(BaseFeatureTransformer):
    def __init__(self, sim, states=('price', ), all_states=False, **kwargs):
        super().__init__(sim=sim, states=states, all_states=all_states)
        self.transformer = PolynomialFeatures(**kwargs)

    def transform(self, X):
        assert X.ndim == 2, X.ndim
        assert max(self.idx) < X.shape[1], (self.states, self.idx)
        return self.transformer.fit_transform(X[:, self.idx])
