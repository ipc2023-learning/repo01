/* Main file, keeps all important variables.
 * Calls functions from "helper_functions" to read in input (variables, operators,
 * goals, initial state),
 * then calls functions to build causal graph, domain_transition_graphs and
 * successor generator
 * finally prints output to file "output"
 */

#include "helper_functions.h"
#include "successor_generator.h"
#include "causal_graph.h"
#include "domain_transition_graph.h"
#include "state.h"
#include "mutex_group.h"
#include "operator.h"
#include "axiom.h"
#include "h2_mutexes.h"
#include "variable.h"
#include "timer.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
using namespace std;

int main(int argc, const char **argv) {
    int h2_mutex_time = 300; // 5 minutes to compute mutexes by default
    bool include_augmented_preconditions = false;
    bool expensive_statistics = false;
    bool disable_bw_h2 = false;
    bool metric;
    vector<Variable *> variables;
    vector<Variable> internal_variables;
    State initial_state;
    vector<pair<Variable *, int>> goals;
    vector<MutexGroup> mutexes;
    vector<Operator> operators;
    vector<Axiom> axioms;
    vector<DomainTransitionGraph> transition_graphs;

    string cost_useful_facts_file;
    vector<string> useful_facts;
    int multiplier_cost_useful_facts = 1;
    int multiplier_cost_semiuseful_facts = 100;
    int multiplier_cost_useless_facts = 10000;

    for (int i = 1; i < argc; ++i) {
        string arg = string(argv[i]);
        if (arg.compare("--no_rel") == 0) {
            cout << "*** do not perform relevance analysis ***" << endl;
            g_do_not_prune_variables = true;
        } else if (arg.compare("--h2_time_limit") == 0) {
            i++;
            if (i < argc) {
                try {
                    h2_mutex_time = atoi(argv[i]);
                }catch (std::invalid_argument &) {
                    cerr << "please specify the number of seconds after --h2_time_limit" << endl;
                    exit(2);
                }
            } else {
                cerr << "please specify the number of seconds after --h2_time_limit" << endl;
                exit(2);
            }
        } else if (arg.compare("--no_h2") == 0) {
            h2_mutex_time = 0;
        } else if (arg.compare("--augmented_pre") == 0) {
            include_augmented_preconditions = true;
        } else if (arg.compare("--no_bw_h2") == 0) {
            disable_bw_h2 = true;
        } else if (arg.compare("--stat") == 0) {
            expensive_statistics = true;
        } else if (arg.compare("--cost-useful-facts") == 0) {
            i++;
            if (i < argc) {
                cost_useful_facts_file = argv[i];
                if (!std::filesystem::exists(cost_useful_facts_file)){
                    cerr << "Error, file does not exist: " << cost_useful_facts_file << endl;
                    exit(2);
                } else {
                    std::ifstream fin(cost_useful_facts_file);
                    std::string str;
                    while (std::getline(fin, str)) {
                        if(str.size() > 0) {
                            useful_facts.push_back(str);
                        }
                    }
                }
            } else {
                cerr << "please specify the file with useful facts after --cost-useful-facts" << endl;
                exit(2);
            }
        } else if (arg.compare("--multiplier-cost-semiuseful-facts") == 0) {
            i++;
            if (i < argc) {
                try {
                    multiplier_cost_semiuseful_facts = atoi(argv[i]);
                }catch (std::invalid_argument &) {
                    cerr << "please specify the multiplier_cost_semiuseful_facts" << endl;
                    exit(2);
                }
            } else {
                cerr << "please specify the multiplier_cost_semiuseful_facts" << endl;
                exit(2);
            }

        } else if (arg.compare("--multiplier-cost-useless-facts") == 0) {
            i++;
            if (i < argc) {
                try {
                    multiplier_cost_useless_facts = atoi(argv[i]);
                }catch (std::invalid_argument &) {
                    cerr << "please specify the multiplier_cost_useless_facts" << endl;
                    exit(2);
                }
            } else {
                cerr << "please specify the multiplier_cost_useless_facts" << endl;
                exit(2);
            }



        } else {
            cerr << "unknown option " << arg << endl << endl;
            cout << "Usage: ./preprocess [--no_rel] [--no_h2]  [--no_bw_h2] [--augmented_pre] [--stat] < output" << endl;
            exit(2);
        }
    }

    read_preprocessed_problem_description
        (cin, metric, internal_variables, variables, mutexes, initial_state, goals, operators, axioms);

    if (!useful_facts.empty()) {
        metric = true;
        vector <pair<Variable *, int>> useful_fact_pairs;
        for (string f : useful_facts) {
            bool found = false;
            for (auto * var : variables) {
                for (int i = 0; i < var->get_range(); ++i) {
                    string fact_name = var->get_fact_name(i);


                    fact_name.erase(std::remove_if(fact_name.begin(), fact_name.end(),[](const unsigned char c)
                                                                                          {
                                                                                              return std::isspace(c);
                                                                                          }
                            ), fact_name.end());

                    std::string::size_type pos = fact_name.find("Atom");
                    if (pos != std::string::npos) {
                        fact_name.erase(pos, 4);
                    }

                    if (fact_name == f) {
                        useful_fact_pairs.push_back(make_pair(var, i));
                        found = true;
                        break;
                    }
                }
                if (found) {
                    break;
                }
            }
            if(!found) {
                cout << "Error: useful fact is not present: " << f << endl;
                exit(-2);
            }
        }

        for (Operator & op : operators) {
            bool some_effect_useful = false;
            bool all_preconditions_useful = true;

            for (auto eff : op.get_prevail()) {
                if (eff.prev >= 0 &&
                    initial_state[eff.var] != eff.prev &&
                    find(useful_fact_pairs.begin(), useful_fact_pairs.end(), make_pair(eff.var, eff.prev)) == useful_fact_pairs.end()) {
                    all_preconditions_useful = false;
                    break;
                }
            }
            for (auto eff : op.get_pre_post()) {
                if (find(useful_fact_pairs.begin(), useful_fact_pairs.end(), make_pair(eff.var, eff.post)) != useful_fact_pairs.end()) {
                    some_effect_useful = true;
                }

                if (eff.pre >= 0 &&
                    initial_state[eff.var] != eff.pre &&
                    find(useful_fact_pairs.begin(), useful_fact_pairs.end(), make_pair(eff.var, eff.pre)) == useful_fact_pairs.end()) {
                    all_preconditions_useful = false;
                }

            }

            bool is_useful = all_preconditions_useful && some_effect_useful;

            if (is_useful) {
                op.multiply_cost(multiplier_cost_useful_facts);
            } else if (some_effect_useful) {
                op.multiply_cost(multiplier_cost_semiuseful_facts);
            } else {
                op.multiply_cost(multiplier_cost_useless_facts);
            }
        }

    }


    //dump_preprocessed_problem_description
    //  (variables, initial_state, goals, operators, axioms);

    cout << "Building causal graph..." << endl;
    CausalGraph causal_graph(variables, operators, axioms, goals);
    const vector<Variable *> &ordering = causal_graph.get_variable_ordering();
    bool cg_acyclic = causal_graph.is_acyclic();

    // Remove unnecessary effects from operators and axioms, then remove
    // operators and axioms without effects.
    strip_mutexes(mutexes);
    strip_operators(operators);
    strip_axioms(axioms);

    // compute h2 mutexes
    if (axioms.size() > 0) {
        cout << "Disabling h2 analysis because it does not currently support axioms" << endl;
    } else if (h2_mutex_time) {
        bool conditional_effects = false;
        for (const Operator &op : operators) {
            if (op.has_conditional_effects()) {
                conditional_effects = true;
                break;
            }
        }
        if (conditional_effects)
            disable_bw_h2 = true;

        if (!compute_h2_mutexes(ordering, operators, axioms,
                                mutexes, initial_state, goals,
                                h2_mutex_time, disable_bw_h2)) {
            // TODO: don't duplicate the code to return an unsolvable task, log and exit here
            cout << "Unsolvable task in preprocessor" << endl;
            generate_unsolvable_cpp_input();
            cout << "Preprocessor time: " << utils::g_timer << endl;
            cout << "Preprocessor peak memory: " << get_peak_memory_in_kb() << " KB" << endl;
            cout << "done" << endl;
            return 0;
        }

        //Update the causal graph and remove unneccessary variables
        strip_mutexes(mutexes);
        strip_operators(operators);
        strip_axioms(axioms);

        cout << "Change id of operators: " << operators.size() << endl;
        // 1) Change id of values in operators and axioms to remove unreachable facts from variables
        for (Operator &op: operators) {
            op.remove_unreachable_facts(ordering);
        }
        // TODO: Activate this if axioms get supported by the h2 heuristic
        // cout << "Change id of axioms: " << axioms.size() << endl;
        // for(int i = 0; i < axioms.size(); ++i){
        //     axioms[i].remove_unreachable_facts();
        // }
        cout << "Change id of mutexes" << endl;
        for (MutexGroup &mutex : mutexes) {
            mutex.remove_unreachable_facts();
        }
        cout << "Change id of goals" << endl;
        vector<pair<Variable *, int>> new_goals;
        for (pair<Variable *, int> &goal : goals) {
            if (goal.first->is_necessary()) {
                goal.second = goal.first->get_new_id(goal.second);
                new_goals.push_back(goal);
            }
        }
        new_goals.swap(goals);
        cout << "Change id of initial state" << endl;
        if (initial_state.remove_unreachable_facts()) {
            // TODO: don't duplicate the code to return an unsolvable task, log and exit here
            cout << "Unsolvable task in preprocessor" << endl;
            generate_unsolvable_cpp_input();
            cout << "Preprocessor time: " << utils::g_timer << endl;
            cout << "Preprocessor peak memory: " << get_peak_memory_in_kb() << " KB" << endl;
            cout << "done" << endl;
            return 0;
        }

        cout << "Remove unreachable facts from variables: " << ordering.size() << endl;
        // 2)Remove unreachable facts from variables
        for (Variable *var : ordering) {
            if (var->is_necessary()) {
                var->remove_unreachable_facts();
            }
        }

        strip_mutexes(mutexes);
        strip_operators(operators);
        strip_axioms(axioms);

        causal_graph.update();
        cg_acyclic = causal_graph.is_acyclic();
        strip_mutexes(mutexes);
        strip_operators(operators);
        strip_axioms(axioms);
    }
    cout << "Building domain transition graphs..." << endl;
    build_DTGs(ordering, operators, axioms, transition_graphs);
    //dump_DTGs(ordering, transition_graphs);
    bool solveable_in_poly_time = false;
    if (cg_acyclic)
        solveable_in_poly_time = are_DTGs_strongly_connected(transition_graphs);
    /*
      TODO: The above test doesn't seem to be quite ok because it
      ignores axioms and it ignores non-unary operators. (Note that the
      causal graph computed here does *not* contain arcs between
      effects, only arcs from preconditions to effects.)

      So solveable_in_poly_time [sic] should also be set to false if
      there are any derived variables or non-unary operators.
     */

    //TODO: genauer machen? (highest level var muss nicht scc sein...gemacht)
    //nur Werte, die wichtig sind fuer drunterliegende vars muessen in scc sein
    cout << "solveable in poly time " << solveable_in_poly_time << endl;
    cout << "Building successor generator..." << endl;
    SuccessorGenerator successor_generator(ordering, operators);
    //successor_generator.dump();

    // Output some task statistics
    int facts = 0;
    int derived_vars = 0;
    for (Variable *var : ordering) {
        facts += var->get_range();
        if (var->is_derived())
            derived_vars++;
    }
    cout << "Preprocessor variables: " << ordering.size() << endl;
    cout << "Preprocessor facts: " << facts << endl;
    cout << "Preprocessor derived variables: " << derived_vars << endl;
    cout << "Preprocessor operators: " << operators.size() << endl;
    cout << "Preprocessor mutex groups: " << mutexes.size() << endl;

    if (expensive_statistics) {
        //Count potential preconditions
        int num_total_augmented = 0;
        int num_op_augmented = 0;
        int num_total_potential = 0;
        int num_op_potential = 0;
        int num_total_potential_noeff = 0;
        int num_op_potential_noeff = 0;

        for (const Operator &op : operators) {
            int count = op.count_augmented_preconditions();
            if (count) {
                num_op_augmented++;
                num_total_augmented += count;
            }
            count = op.count_potential_preconditions();
            if (count) {
                num_op_potential++;
                num_total_potential += count;
            }
            count = op.count_potential_noeff_preconditions();
            if (count) {
                num_op_potential_noeff++;
                num_total_potential_noeff += count;
            }
        }

        cout << "Augmented preconditions: " << num_total_augmented << endl;
        cout << "Ops with augmented preconditions: " << num_op_augmented << endl;
        cout << "Potential preconditions: " << num_total_potential << endl;
        cout << "Ops with potential preconditions: " << num_op_potential << endl;
        cout << "Potential preconditions contradict effects: " << num_total_potential_noeff << endl;
        cout << "Ops with potential preconditions contradict effects: " << num_op_potential_noeff << endl;
        set<vector<int>> mutexes_fw, mutexes_bw;
        for (MutexGroup &mutex : mutexes) {
            if (!mutex.is_redundant()) {
                if (mutex.is_fw())
                    mutex.add_tuples(mutexes_fw);
                else
                    mutex.add_tuples(mutexes_bw);
            }
        }
        cout << "Preprocessor mutex groups fw: " << mutexes_fw.size()
             << " bw: " << mutexes_bw.size() << endl;
    }

    if (include_augmented_preconditions) {
        for (Operator &op : operators) {
            op.include_augmented_preconditions();
        }
    }
    // Calculate the problem size
    int task_size = ordering.size() + facts + goals.size();

    for (const MutexGroup &mutex : mutexes)
        task_size += mutex.get_encoding_size();

    for (const Operator &op : operators)
        task_size += op.get_encoding_size();

    for (const Axiom &axiom : axioms)
        task_size += axiom.get_encoding_size();

    cout << "Preprocessor task size: " << task_size << endl;

    cout << "Writing output..." << endl;
    if (ordering.empty()) {
        cout << "Unsolvable task in preprocessor" << endl;
        generate_unsolvable_cpp_input();
    } else {
        generate_cpp_input(
            solveable_in_poly_time, ordering, metric,
            mutexes, initial_state, goals,
            operators, axioms, successor_generator,
            transition_graphs, causal_graph);
    }
    cout << "Preprocessor time: " << utils::g_timer << endl;
    cout << "Preprocessor peak memory: " << get_peak_memory_in_kb() << " KB" << endl;
    cout << "done" << endl;
}
