import numpy as np


# Load a profile file - wrapper for load_prf2

def load_prf(filename):
    psi, var = load_prf2(filename)
    return psi, var


# Load a profile file.
# Returns psi, var

def load_prf2(filename):
    with open(filename, 'r') as file:
        [n] = map(int, file.readline().strip().split())
        psi = np.zeros(n)
        var = np.zeros(n)

        for l in range(n):
            [psi[l], var[l]] = map(float, file.readline().strip().split())

        [end_flag] = map(int, file.readline().strip().split())
        if end_flag != -1:
            print('Error: end flag is not -1. end_flag= %d' % end_flag)
        return psi, var


# Save a profile file

def save_prf(x, y, fname):
    with open(fname, "w") as f:
        sz = np.size(x)
        f.write("%d\n" % sz)
        for i in range(sz):
            f.write("%19.13e  %19.13e\n" % (x[i], y[i]))
        f.write("-1\n")


# three functions related with Kinetic EFIT p-file

def read_kefit_profile(pfilename):
    profiles = []
    ion_species = None

    with open(pfilename, 'r') as file:
        lines = file.readlines()

    line_index = 0

    while line_index < len(lines):
        line = lines[line_index].strip()
        if not line:
            line_index += 1
            continue

        try:
            parts = line.split()
            if len(parts) >= 1 and parts[0].isdigit():
                num_points = int(parts[0])

                if len(parts) >= 5 and parts[1] == "N" and parts[2] == "Z" and parts[3] == "A":
                    ion_species = {
                        'count': num_points,
                        'description': ' '.join(parts[1:]),
                    }

                    species_data = np.zeros((num_points, 3))
                    for i in range(num_points):
                        if line_index + 1 + i < len(lines):
                            data_line = lines[line_index + 1 + i].strip().split()
                            if len(data_line) >= 3:
                                species_data[i, 0] = float(data_line[0])
                                species_data[i, 1] = float(data_line[1])
                                species_data[i, 2] = float(data_line[2])

                    ion_species['data'] = species_data
                    line_index += num_points + 1

                else:
                    profile_info = {'num_points': num_points}

                    if len(parts) >= 3:
                        profile_info['name'] = parts[1]

                        second_part = parts[2]
                        if '(' in second_part and ')' in second_part:
                            name, units = second_part.split('(')
                            units = units.split(')')[0]
                            profile_info['name'] = name
                            profile_info['units'] = units
                        else:
                            profile_info['name'] = second_part
                            profile_info['units'] = ''

                        if len(parts) >= 4:
                            profile_info['gradient_label'] = parts[3]
                        else:
                            profile_info['gradient_label'] = ''

                    data = np.zeros((3, num_points))
                    for i in range(num_points):
                        if line_index + 1 + i < len(lines):
                            data_line = lines[line_index + 1 + i].strip().split()
                            if len(data_line) >= 3:
                                data[0, i] = float(data_line[0])
                                data[1, i] = float(data_line[1])
                                data[2, i] = float(data_line[2])

                    profile_info['data'] = data
                    profiles.append(profile_info)

                    line_index += num_points + 1
            else:
                line_index += 1
        except Exception as e:
            print(f"Error parsing line {line_index}: {e}")
            line_index += 1

    return profiles, ion_species


def plot_profiles(profiles):
    try:
        import matplotlib.pyplot as plt

        for i, profile in enumerate(profiles):
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(profile['data'][0], profile['data'][1])
            plt.xlabel('psinorm')
            plt.ylabel(f"{profile['name']} ({profile['units']})")
            plt.title(f"Profile {i+1}: {profile['name']}")
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(profile['data'][0], profile['data'][2])
            plt.xlabel('psinorm')
            plt.ylabel(profile['gradient_label'])
            plt.title(f"Gradient of {profile['name']}")
            plt.grid(True)

            plt.tight_layout()
            plt.show()
    except ImportError:
        print("Matplotlib is not installed. Cannot plot profiles.")


def display_ion_species(ion_species):
    if ion_species is None:
        print("No ion species information in file.")
        return

    print(f"Ion Species Information ({ion_species['description']}):")
    print(f"Number of species: {ion_species['count']}")

    print("    N      Z      A")
    print("----------------------")
    for i in range(ion_species['count']):
        print(f"{ion_species['data'][i, 0]:6.2f} {ion_species['data'][i, 1]:6.2f} {ion_species['data'][i, 2]:10.6f}")
