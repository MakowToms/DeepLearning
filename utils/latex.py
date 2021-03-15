import statistics


def add_separator(f, condition):
    if condition:
        f.write(" \\\\\n")
    else:
        f.write(" & ")


def save_activation_error_to_latex(activation_error, activations, metrics, filename="temp.txt", shrink=False):
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

# \begin{tabular}{ |p{3cm}||p{3cm}|p{3cm}|p{3cm}|  }
#  \hline
#  \multicolumn{4}{|c|}{Country List} \\
#  \hline
#  Country Name     or Area Name& ISO ALPHA 2 Code &ISO ALPHA 3 Code&ISO numeric Code\\
#  \hline
#  Afghanistan   & AF    &AFG&   004\\
#  Aland Islands&   AX  & ALA   &248\\
#  Albania &AL & ALB&  008\\
#  Algeria    &DZ & DZA&  012\\
#  American Samoa&   AS  & ASM&016\\
#  Andorra& AD  & AND   &020\\
#  Angola& AO  & AGO&024\\
#  \hline
# \end{tabular}
