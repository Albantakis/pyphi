from ..models import Part, Bipartition
from ..partition import bipartition_of_one, partition_types, bipartition, directed_bipartition
from itertools import product

@partition_types.register('ONE_MECH')
def one_mech_partitions(mechanism, purview, node_labels=None):
    # cut one mechanism unit and none or one purview unit
    # given the way purviews are selected, each mechanism node has a particular purview node with 
    # which it is most correlated, which corresponds to itself the way the tpm is loaded, so we either cut no purview
    # or the one purview node with largest correlation
    if len(mechanism) < 2:
        yield Bipartition(
            Part(tuple(mechanism), ()), Part((), tuple(purview))
        )
    else:
        numerators = bipartition_of_one(mechanism)
        for n in numerators:
            if len(set(n[0]) & set(purview)) > 0:
                yield Bipartition(
                    Part(n[0], n[0]), Part(n[1], tuple(set(n[1]) & set(purview))), node_labels=node_labels
                )

            yield Bipartition(
                Part(n[0], tuple([])), Part(n[1], tuple(purview)), node_labels=node_labels
            )