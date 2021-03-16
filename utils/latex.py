import statistics


def add_separator(f, condition):
    if condition:
        f.write(" \\\\\n")
    else:
        f.write(" & ")


def save_errors_to_latex(errors, labels, metrics, caption, tex_label, filename="temp.txt", shrink=False):
    with open(filename, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\caption{" + caption + "}\n")
        f.write("\\label{" + tex_label + "}\n")

        f.write(r"\begin{tabularx}{\textwidth}{ |X|X|X| }")
        f.write("\n\\hline\n")

        f.write(r"Measure & Activation & Value \\")
        f.write("\n\\hline\n")

        for met_index, metric in enumerate(metrics):
            f.write(r"\multirow{{{0}}}*{{{1}}} & ".format(len(labels), metric))
            for act_index, label in enumerate(labels):
                if act_index != 0:
                    f.write(r"& ")
                f.write(r"{0} & ".format(label))
                mean = statistics.mean(errors[act_index][met_index])
                std = statistics.stdev(errors[act_index][met_index])
                f.write(r"${0:.3f} \pm {1:.3f}$ \\".format(mean, std))
                f.write("\n")
            f.write("\\hline\n")
        f.write("\\end{tabularx}\n")
        f.write("\\end{table}\n")


def save_activation_error_to_latex(activation_error, activations, metrics, filename="temp.txt", shrink=False):
    with open(filename, "w") as f:
        f.write(r"\begin{tabularx}{\textwidth}{ |X|X|X| }")
        f.write("\n\\hline\n")

        f.write(r"Measure & Activation & Value \\")
        f.write("\n\\hline\n")

        for met_index, metric in enumerate(metrics):
            f.write(r"\multirow{{{0}}}*{{{1}}} & ".format(len(activations), metric))
            for act_index, activation in enumerate(activations):
                if act_index != 0:
                    f.write(r"& ")
                f.write(r"{0} & ".format(activation.name))
                mean = statistics.mean(activation_error[act_index][met_index])
                std = statistics.stdev(activation_error[act_index][met_index])
                f.write(r"${0:.3f} \pm {1:.3f}$ \\".format(mean, std))
                f.write("\n")
            f.write("\\hline\n")
        f.write("\\end{tabularx}\n")


def save_activation_error_to_latex_wide(activation_error, activations, metrics, filename="temp.txt", shrink=False):
    with open(filename, "w") as f:
        f.write(r"\begin{tabularx}{\textwidth}{ |" + "X|" * len(activations) * len(metrics) + " }\n")
        f.write("\\hline\n")

        # headers
        for i, metric in enumerate(metrics):
            f.write("\\multicolumn{{{0}}}{{|c|}}{{{1}}}".format(
                len(activations), metric))
            add_separator(f, i == len(metrics) - 1)
        f.write("\\hline\n")

        for i, activation in enumerate(activations * len(metrics)):
            if shrink:
                f.write("\\footnotesize{{{0}}}".format(activation.name))
            else:
                f.write(activation.name)
            add_separator(f, i == len(metrics) * len(activations) - 1)
        f.write("\\hline\n")

        # values
        for i in range(len(activation_error[0][0])):
            for met_index, metric in enumerate(metrics):
                for act_index, activation in enumerate(activations):
                    if shrink:
                        f.write("\\footnotesize{{{:.3f}}}".format(activation_error[act_index][met_index][i]))
                    else:
                        f.write("{:.3f}".format(activation_error[act_index][met_index][i]))
                    add_separator(f, (act_index == len(activations) - 1) and (met_index == len(metrics) - 1))
        f.write("\\hline\n")

        # means
        for met_index, metric in enumerate(metrics):
            for act_index, activation in enumerate(activations):
                value = statistics.mean(activation_error[act_index][met_index])
                if shrink:
                    f.write("\\footnotesize{{{:.3f}}}".format(value))
                else:
                    f.write("{:.3f}".format(value))
                add_separator(f, (act_index == len(activations) - 1) and (met_index == len(metrics) - 1))
        f.write("\\hline\n")

        # standard deviations
        for met_index, metric in enumerate(metrics):
            for act_index, activation in enumerate(activations):
                value = statistics.stdev(activation_error[act_index][met_index])
                if shrink:
                    f.write("\\footnotesize{{{:.3f}}}".format(value))
                else:
                    f.write("{:.3f}".format(value))
                add_separator(f, (act_index == len(activations) - 1) and (met_index == len(metrics) - 1))
        f.write("\\hline\n")
        f.write("\\end{tabularx}\n")
