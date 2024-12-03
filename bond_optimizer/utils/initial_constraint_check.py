from bond_optimizer.utils.helpers import filter_dataset, scope_map

# Performs an initial trivial check for all constraints to ensure they are feasible. 
# For example, if the portfolio has a total of 5 bonds, and we set a lowerbound cardinality of 10, 
# that is immediately infeasible.

def check_constraint(dataset, benchmark_metrics, metric, lowerbound, upperbound):
    ''' The first two checks were not implemented. '''
    match metric:
        # For DTS/Duration, a trivial constraint is that the upperbound constraint must be greater than the value of the minimum DTS/Duration of any asset
        
        # case 'DTS':
        #     minDTS = (dataset['OAS'] * dataset['DURATION']).min()
        #     if benchmark_metrics['total_dts'] + upperbound < minDTS:
        #         return False
        # case 'Duration':
        #     minDuration = dataset['DURATION'].min()
        #     if benchmark_metrics['total_duration'] + upperbound < minDuration:
        #         return False

        # For Cardinality, we must have a lowerbound which is lower than the number of bonds in the dataset
        case 'Cardinality':
            num_bonds = len(dataset)
            if lowerbound > num_bonds:
                return False
    return True

def initial_constraint_checks(dataset, constraints, benchmark_metrics):
    '''Perform some initial checks for constraints which might be impossible.
    For example, it is definitely ipmossible to get a minimum of 5 cardinality
    for all sectors if we only have 3 bonds in the 'Finance' sector'''
    violated_constraints = []
    if len(dataset) == 0:
        return []
    for constraint in constraints:
        metric = constraint.metric
        scope = constraint.scope
        bucket = constraint.bucket
        lowerbound = constraint.lowerbound
        upperbound = constraint.upperbound

        if bucket == 'ALL':
            scope_key = scope_map(scope)
            buckets = set(dataset[scope_key].unique())

            for bucket in buckets:
                filtered_dataset = filter_dataset(dataset, scope, bucket)
                if not check_constraint(filtered_dataset, benchmark_metrics, metric, lowerbound, upperbound):
                    violated_constraints.append(constraint)
                    break
        else:
            # Filter data based on scope
            filtered_dataset = filter_dataset(dataset, scope, bucket)
            if not check_constraint(filtered_dataset, benchmark_metrics, metric, lowerbound, upperbound):
                violated_constraints.append(constraint)

    return violated_constraints