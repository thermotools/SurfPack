'''
Functions to auto-generate markdown documentation by parsing docstrings

Intent : As long as we keep docstrings consistently formatted, this ensures that we can quickly generate up-to-date
        documentation that is easily accessible. It also ensures that documentation only needs to be updated one place,
        and that place is in the codebase itself.

Usage : Current functionality is designed to parse the docstrings of a given class. It assumes that the docstrings of
        the methods in the class are formatted as

        def myfunc(self, p1, p2, p3, p4=None, p5=<something>, ...):
            """Section Name
            Description of what this function does (a kind of header). We can write lots of stuff here
            NOTE the double lineshift here

            Args:
                 p1 (int) : The lineshift before 'Args' is necessary.
                 p2 (float) : The colons here are also necessary.
                 p3 (bool) : SomethingSomething
                 p4 (Optional, list) : etc.
            Returns:
                (float) : The colon here is also necessary.
            """
-----------------------------------------------------------------------------------------------------------------------

NOTE: The procedure described below is the general procedure for generating documentation for large classes
        with many methods doing different things. For simple classes, consider using the function
        `basic_class_to_markdown` to do most of the job.

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
Now for the generic procedure:

    1 :  Use the function get_autogen_header(<classname>) to generate a comment with a timestamp and a description of
         how the markdown file was generated (using this module)

    2 : Define a list of sections that the methods in the class should be divided into, see : thermo_to_markdown() for
        an example. The sections should equate to the 'Section Name' in the above formatting example. The sections in
        the markdown file will be ordered according to the order of this list.

        NOTE: The sections "Other" and "Deprecated" are automatically generated.
            If the section name in a methods docstring does not match any of the supplied sections, it is added to the
            "Other" section, and a warning is issued
            If a method has the Section Name "Deprecated" it is added to the automatically generated "Deprecated" section.
            To add other automatically generated sections, add them in the function `get_automatic_sections`

        NOTE: A method can be added to several sections, by separating the "Section Name"'s with an ampersand as
            def my_func(self):
                """Section 1 & Section 5 & Section 6 & ...
                This is documentation for a method that belongs to many sections
                """

    3 : Define a dict of section name => section header mappings. This maps the section names in the docstrings to
        the section headers in the markdown file.

    4 : Define a dict of section name => section intro mappings. This maps the section names to an intro text that is
        displayed at the start of the section.

    5 : Use
            my_methods = inspect.getmembers(<MyClass>, predicate=inspect.isfunction)
        to extract the list of class methods.

    6 : Use
            method_dict = split_methods_by_section(sections, my_methods)
        to split the methods into a single list for each method, placed in a dict

    7 : Use
            toc_markdown_text = get_toc(sections, section_headers, method_dict)
        to generate a string, consisting of the markdown table of contents (with links)

    8 : Use
            contents = get_markdown_contents(sections, section_headers, section_intro, method_dict)
        to generate the contents of the file

    9 : Concatenate the strings you have generated and use `write_file` to write to file.
    Note: USE The method `write_file`: It checks for content changes and does not update the file if no
        other content than the timestamp has changed. This prevents you from needing to push a bunch of changes that
        are only timestamp changes, and makes our history much more clean.
'''
import copy
import inspect
from warnings import warn
from datetime import datetime
from surfpack.Functional import Functional
from surfpack.saft import SAFT
from surfpack.pcsaft import PC_SAFT
from thermopack.thermo import thermo
from thermopack.saft import saft
from thermopack.pets import pets
from thermopack.cubic import cubic
from tools import remove_illegal_link_chars, write_file, MARKDOWN_DIR


def get_autogen_header(classname):
    header = f'''---
layout: default
version: 
title: Methods in the {classname} class
permalink: /vcurrent/{classname}_methods.html
---\n\n'''
    header += '<!--- \n'
    header += 'Generated at: ' + datetime.today().isoformat() + '\n'
    header += 'This is an auto-generated file, generated using the script at ' \
              'surfpack/docs/tools/markdown_from_docstrings.py\n'
    header += 'The file is created by parsing the docstrings of the methods in the \n'
    header += classname + ' class. For instructions on how to use the parser routines, see the\n' \
                          'file surfpack/docs/tools/markdown_from_docstrings.py'
    header += '--->\n\n'
    return header


def to_markdown(methods):
    """
    Generate markdown text from the docstring of the methods in the list methods

    Args:
        methods list<tuple<str, function>> : A list of tuples, with the first element being the name of the method,
                                            and the second element being the method itself.
    Returns:
        (str) : A markdown - formatted string with the contents of the docstring.
    """
    md_text = ''
    for name, meth in methods:
        docparts = meth.__doc__.split('\n\n')
        header_lines = [line.strip() for line in docparts[0].split('\n')]
        header = ' '.join(header_lines[1:]) # Cutting out the section identifier
        header.replace('\n', ' ')

        if len(docparts) > 1:
            content = '\n\n'.join(docparts[1:])
        else:
            content = ''

        content_lines = [line.strip() for line in content.split('\n')]

        md_text += '### `' + name + str(inspect.signature(meth)) + '`\n'
        md_text += header + '\n\n'

        pad = '&nbsp;' * 4 + ' '
        endl = '\n\n'
        for line in content_lines:
            if ('args:' in line.lower()) or ('returns:' in line.lower()) or ('raises:' in line.lower()):
                md_text += '#### ' + line + endl

            elif ':' in line:
                line = line.split(':')
                md_text += pad + '**' + line[0] + ':** ' + endl
                md_text += 2 * pad + ':'.join(line[1:]) + endl
            else:
                md_text += 2 * pad + line + endl

    return md_text

def split_methods_by_section(sections, methods):
    """
    Organise the various methods of a class into sections, determined by the id 'Section Name' as in the example at
    the top of this file. Warns if there are methods for which no matching section was found.
    The section 'Other' is automatically generated if there are methods with an identifier not matching any of the
    identifiers in 'sections'. The section 'Deprecated' is automatically included in the 'sections' list.

    Args:
        sections (list<str>) : The name of each section, corresponding to the first line in each docstring.
        methods (list<tuple<str, function>>) : List of tuples, with the first element corresponding to the method
                                                names, and the second to the methods themself.
    Returns:
        dict : With keys corresponding to the sections, and value being a list of the (name, method) tuples with
                the first line of the docstring matching the section (key).
    """

    if 'deprecated' not in [s.lower() for s in sections]:
        sections.append('Deprecated')

    method_dict = {}
    for name, meth in methods:
        # A method can be added to several sections, by giving it the header
        # def myfunc():
        #   """Section1 & Section2 & Section5 ... """
        method_sections = [m.strip() for m in meth.__doc__.split('\n')[0].lower().split('&')] # extracting the section names as described above
        method_has_section = False
        for sec in sections:
            if sec.lower() in method_sections:
                method_has_section = True
                if sec in method_dict.keys():
                    method_dict[sec].append((name, meth))
                else:
                    method_dict[sec] = [(name, meth)]
        if method_has_section is False: # Generate a section called "Other" and add the method to that section. Warn that this is done.
            warn('Method : ' + name + " did not contain a section on the first line of its docstring! Adding to 'Other'",
                 SyntaxWarning, stacklevel=3)
            warn(f'Section names in docstring are : {method_sections}', SyntaxWarning, stacklevel=3)
            warn(f'Available sections are : {sections}', SyntaxWarning, stacklevel=3)

            if 'Other' in method_dict.keys():
                method_dict['Other'].append((name, meth))
            else:
                method_dict['Other'] = [(name, meth)]

        # print('other is : ', method_dict['Other'] if 'Other' in method_dict.keys() else None)
    return method_dict

def get_automatic_sections(sections, section_headers, section_intro, method_dict):
    sections = copy.deepcopy(sections)
    section_headers = copy.deepcopy(section_headers)
    section_intro = copy.deepcopy(section_intro)

    if 'Other' in method_dict.keys():
        if 'Other' not in sections:
            sections.append('Other')
        if 'Other' not in section_headers.keys():
            section_headers['Other'] = 'Other'
        if 'Other' not in section_intro.keys():
            section_intro['Other'] = 'Methods that do not have a section identifier in their docstring.'

    if 'Deprecated' in method_dict.keys():
        if 'Deprecated' not in sections:
            sections.append('Deprecated')
        if 'Deprecated' not in section_headers.keys():
            section_headers['Deprecated'] = 'Deprecated methods'
        if 'Deprecated' not in section_intro.keys():
            section_intro['Deprecated'] = 'Deprecated methods are not maintained, and may be removed in the future.'

    return sections, section_headers, section_intro

def get_toc(sections, section_headers, method_dict, is_subsection=False):
    """
    Generate a table of contents with links to the sections, using the names in section_headers.
    Note: Uses the function tools.remove_illegal_link_chars() to sanitize raw strings, such that they can be used
    in markdown links.

    Args:
        sections (list<str>) : List of the section names, corresponding to the first line in the docstrings
        section_headers (dict<str, str>) : Dict mapping the section names to the section headers in the markdown file
        method_dict (dict<str, list<tuple<str, function>>>) : Mapping the section names to the list of (method name,
                                                                function) tuples that are in the section.
        is_subsection (bool, optional) : Whether to include the automatically generated sections in the toc.
                                        Should be set to False when generating the toc for the whole file, and set to
                                        True when generating toc for subsections.
    Returns:
        str : The string representation of the table of contents to be written to the markdown file.
    """

    if is_subsection is False:
        sections, section_headers, _ = get_automatic_sections(sections, section_headers, {}, method_dict)

    toc_text = '## Table of contents\n'
    for sec in sections:
        if sec not in method_dict.keys():
            continue

        sec_name = section_headers[sec]
        sec_id = remove_illegal_link_chars(sec_name)
        toc_text += '  * [' + sec_name + '](#' + sec_id + ')\n'

        for meth in method_dict[sec]:
            method_name = meth[0].replace('__', '\_\_')
            method_id = meth[0] + str(inspect.signature(meth[1]))
            method_id = remove_illegal_link_chars(method_id)
            toc_text += '    * [' + method_name + '](#' + method_id + ')\n'

    return toc_text + '\n'

def get_markdown_contents(sections, section_headers, section_intro, method_dict, sub_toc=True):
    """
    Iterate through the sections, generate the markdown documentation for all methods in method_dict, and join them
    while adding section headers and section introductions

    Args:
        sections (list<str>) : List of the section names, corresponding to the first line in the docstrings
        section_headers (dict<str, str>) : Dict mapping the section names to the section headers in the markdown file
        section_intro (dict<str, str>) : Dict mapping the section names to the section introduction text.
        method_dict (dict<str, list<tuple<str, function>>>) : Mapping the section names to the list of (method name,
                                                                function) tuples that are in the section.
        sub_toc (bool) : Whether to add ToC at the top of each sub-section.
    Returns:
        str : The markdown text corresponding to the main contents of the file.
    """

    sections, section_headers, section_intro = get_automatic_sections(sections, section_headers, section_intro, method_dict)

    md_text = ''
    for sec in sections:
        if sec not in method_dict.keys():
            continue
        md_text += '## ' + section_headers[sec] + '\n\n'
        md_text += section_intro[sec] + '\n\n'
        if sub_toc is True:
            md_text +=  '#' + get_toc([sec], section_headers, method_dict, is_subsection=True) + '\n'
        md_text += to_markdown(method_dict[sec])

    return md_text

def basic_class_to_markdown(classname, eosname, methods, intro_text=None, inherits=None):
    """
    Generate markdown documentation file for a class that implements only Constructor and unility methods.
    """

    sections = ['Constructor',
                'Utility']

    section_headers = {'Constructor': 'Constructor',
                       'Utility': 'Utility methods'}

    section_intro = {'Constructor': f'Methods to initialise {eosname} model.',
                     'Utility': 'Set- and get methods for interaction parameters, mixing parameters ...'}

    method_dict = split_methods_by_section(sections, methods)

    ofile_text = get_autogen_header(classname)
    if intro_text is None:
        if inherits is None:
            ofile_text += f'The `{classname}` class, found in `addon/pycThermopack/thermopack/{classname}.py`, is the interface to the \n' \
                          f'{eosname} Equation of State. This class implements utility methods to access mixing parameters etc.\n\n'
        else:
            ofile_text += f'The `{classname}` class, found in `addon/pycThermopack/thermopack/{classname}.py`, inherrits ' \
                          f'from the {inherits} class, and  is the interface to the \n' \
                          f'{eosname} Equation of State. This class implements utility methods to access mixing parameters etc.\n\n'
    else:
        ofile_text += intro_text
    ofile_text += get_toc(sections, section_headers, method_dict)
    ofile_text += get_markdown_contents(sections, section_headers, section_intro, method_dict)

    filename = f'{classname}_methods.md'
    write_file(MARKDOWN_DIR + filename, ofile_text)

def Functional_to_markdown():
    """
    Generate markdown documentation file for the Functional class.
    """

    sections = ['Profile Property',
                'rhoT Property',
                'Bulk Property',
                'Pure Property',
                'Weights',
                'Weighted density',
                'Density Profile',
                'Helmholtz Contribution',
                'Utility',
                'Internal']

    section_headers = {'Internal' : 'Internal methods',
                       'Utility' : 'Utility methods',
                       'Profile Property' : 'Profile property interfaces',
                       'rhoT Property' : r'$\rho - T$ property interfaces',
                       'Bulk Property' : 'Bulk property interfaces',
                       'Pure Property' : 'Pure fluid properties',
                       'Weights' : 'Weight function interfaces',
                       'Weighted density' : 'Weighted density computations',
                       'Density Profile' : 'Density profile computations',
                       'Helmholtz Contribution' : 'Helmholtz Contributions'}

    section_intro = {'Internal': 'Methods for handling communication with the Fortran library.',
                       'Utility': 'Methods for setting ... and getting ...',
                       'Profile Property' : 'Compute properties using a given density profile. Note that properties computed\n'
                                            'using these methods do not generally check whether the Profile is a valid\n'
                                            'equilibrium Profile. Properties are computed from the Profile ''as is''.\n'
                                            'For methods to compute equilibrium density profiles, see the Density Profile\n'
                                            'section. For methods that implicitly compute the density profile for given\n'
                                            'boundary conditions before computing a property see ''rho-T properties''.',
                       'rhoT Property' : 'Compute properties at a given density and temperature, ususally by first '
                                         'computing a density profile for the given state.',
                       'Bulk Property' : 'Evaluating bulk properties.',
                       'Pure Property' : 'Methods to efficiently and conveninetly compute properties for pure fluids. Contain\n'
                                         'some optimisations, tuning and convenience factors that are only possible for\n'
                                         'pure fluids.',
                       'Weights' : 'Get-methods for weight functions.',
                       'Weighted density' : 'Compute weighted densities',
                       'Density Profile' : 'Methods for converging a density profile given various boundary conditions.',
                       'Helmholtz Contribution' : 'Compute different contributions to the reduced Helmholtz energy density.'}

    thermo_methods = inspect.getmembers(Functional, predicate=inspect.isfunction)
    method_dict = split_methods_by_section(sections, thermo_methods)

    ofile_text = get_autogen_header('Functional')
    ofile_text += 'The `Functional` class, found in `surfpack/Functional.py`, is the core of SurfPack Density Functional\n' \
                  'Theory. This is the interface to almost all practical computations, such as computation of surface\n' \
                  'tensions, adsorbtion, etc. and also contains the interfaces to methods used for computing density\n' \
                  'profiles in systems with various boundary conditions. All DFT models in SurfPack inherit the `Functional`\n' \
                  'class.\n\n'
    ofile_text += get_toc(sections, section_headers, method_dict)
    ofile_text += get_markdown_contents(sections, section_headers, section_intro, method_dict)

    filename = 'Functional_methods.md'
    write_file(MARKDOWN_DIR + filename, ofile_text)

def saft_to_markdown():
    """
    Generate markdown documentation file for the saft class.
    """

    sections = ['Utility',
                'Internal',
                'Profile Property',
                'Helmholtz Contribution',
                'Weighted density',
                'Weights']

    section_headers = {'Internal' : 'Internal methods',
                       'Utility' : 'Utility methods',
                       'Profile Property' : 'Profile property interfaces',
                       'Helmholtz Contribution' : 'Helmholtz energy contributions',
                       'Weighted density' : 'Weighted densities',
                       'Weights' : 'Weight functions'}

    section_intro = {'Internal': 'Internal use, of little interest to outside users.',
                       'Utility': 'Methods for computing specific parameters and contributions to the residual\n'
                                  'Helmholtz energy for SAFT-type equations of state',
                     'Profile Property': 'Compute properties using a given density profile, without iterating the density\n'
                                         'profile to ensure equilibrium.',
                     'Helmholtz Contribution' : 'Methods for computing various contributions to the Helmholtz energy\n'
                                                 'that are present in all SAFT-based functionals.',
                     'Weighted density' : 'Methods to compute various weighted densities required by the different\n'
                                          'Helmholtz energy contributions.',
                     'Weights' : 'Get-methods for various weight functions.'}

    saft_methods = inspect.getmembers(SAFT, predicate=inspect.isfunction)
    parent_methods = inspect.getmembers(Functional, predicate=inspect.isfunction)
    saft_specific_methods = sorted(list(set(saft_methods) - set(parent_methods)))
    method_dict = split_methods_by_section(sections, saft_specific_methods)

    ofile_text = get_autogen_header('saft')
    ofile_text += 'The `SAFT` class, found in `surfpack/saft.py`, is an abstract class, that is inherited\n' \
                  'by the `SAFT_VR_Mie` and `PC_SAFT` classes. It contains some generic utility methods to\n' \
                  'compute quantities of interest when investigating SAFT-type functionals.\n\n'
    ofile_text += get_toc(sections, section_headers, method_dict)
    ofile_text += get_markdown_contents(sections, section_headers, section_intro, method_dict)

    filename = 'SAFT_methods.md'
    write_file(MARKDOWN_DIR + filename, ofile_text)

def pcsaft_to_markdown():
    """
    Generate markdown documentation file for the pcsaft class.
    """

    sections = ['Constructor',
                'Utility',
                'Internal',
                'Profile Property',
                'Helmholtz Contribution',
                'Weighted density',
                'Weights']

    section_headers = {'Constructor' : 'Constructor',
                       'Internal' : 'Internal methods',
                       'Utility' : 'Utility methods',
                       'Profile Property' : 'Profile property interfaces',
                       'Helmholtz Contribution' : 'Helmholtz energy contributions',
                       'Weighted density' : 'Weighted densities',
                       'Weights' : 'Weight functions'}

    section_intro = {'Constructor' : 'Construction method(s).',
                     'Internal': 'Internal use, mainly documented for maintainers and developers.',
                     'Utility': 'Methods for computing specific parameters and contributions to the residual\n'
                                  'Helmholtz energy for PC-SAFT-type equations of state',
                     'Profile Property': 'Compute properties using a given density profile, without iterating the density\n'
                                         'profile to ensure equilibrium.',
                     'Helmholtz Contribution' : 'Methods for computing various contributions to the Helmholtz energy\n'
                                                 'that are present in all SAFT-based functionals.',
                     'Weighted density' : 'Methods to compute various weighted densities required by the different\n'
                                          'Helmholtz energy contributions.',
                     'Weights' : 'Get-methods for various weight functions.'}

    saft_methods = inspect.getmembers(PC_SAFT, predicate=inspect.isfunction)
    parent_methods = inspect.getmembers(SAFT, predicate=inspect.isfunction)
    saft_specific_methods = sorted(list(set(saft_methods) - set(parent_methods)))
    method_dict = split_methods_by_section(sections, saft_specific_methods)

    ofile_text = get_autogen_header('saft')
    ofile_text += 'The `PC_SAFT` class, found in `surfpack/pcsaft.py`, inherits the `SAFT` class and implements several\n' \
                  'contributions to the Helmholtz energy density, such as association and chain contributions.\n\n'
    ofile_text += get_toc(sections, section_headers, method_dict)
    ofile_text += get_markdown_contents(sections, section_headers, section_intro, method_dict)

    filename = 'PC_SAFT_methods.md'
    write_file(MARKDOWN_DIR + filename, ofile_text)

def pets_to_markdown():
    classname = 'pets'
    eosname = 'PeTS'
    inherits = 'saft'

    class_methods = inspect.getmembers(pets, predicate=inspect.isfunction)
    parent_methods = inspect.getmembers(saft, predicate=inspect.isfunction)
    specific_methods = sorted(list(set(class_methods) - set(parent_methods)))

    basic_class_to_markdown(classname, eosname, specific_methods, inherits=inherits)

def cubic_to_markdown():

    classname = 'cubic'
    eosname = 'Cubic'

    with open(MARKDOWN_DIR + 'cubic_keys.md', 'r') as intro_file:
        intro_text = intro_file.read()

    from thermopack.cubic import SoaveRedlichKwong, RedlichKwong, VanDerWaals, PengRobinson, PengRobinson78, PatelTeja, SchmidtWensel
    parent_methods = inspect.getmembers(cubic, predicate=inspect.isfunction)
    child_sections = ['Constructor']

    child_section_headers = {'Constructor': 'Constructor'}
    child_section_intro = {'Constructor': f'Methods to initialise model.'}
    child_method_dict = {}
    classes = [SoaveRedlichKwong, RedlichKwong, VanDerWaals, PengRobinson, PengRobinson78, PatelTeja, SchmidtWensel]
    for cls in classes:
        child_sections.append(cls.__name__)
        child_section_headers[cls.__name__] = cls.__name__
        child_section_intro[cls.__name__] = f'Interface to the `{cls.__name__}` EoS'
        class_methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        child_method_dict[cls.__name__] = sorted(list(set(class_methods) - set(parent_methods)))

    ofile_text = get_autogen_header(classname)
    ofile_text += f'The `{classname}` class, found in `addon/pycThermopack/thermopack/{classname}.py`, is the interface to the \n' \
                    f'{eosname} Equation of State. This class implements utility methods to access mixing parameters etc.\n\n'
    ofile_text += f'In addition to the `cubic` class, there are several convenience classes to give easy access to ' \
                  f'specific cubic equations of state. The sections [Initialiser keys](#initialiser-keys), [Pure fluid &alpha;](#pure-fluid-&alpha;), ' \
                  f'[&alpha; mixing rules](#&alpha;-mixing-rules) and [&beta; mixing rules](#&beta;-mixing-rules) ' \
                  f'summarise the various valid input keys that can be used to modify mixing rules, the &alpha; -parameter ' \
                  f'and the underlying EoS.\n\nDocumentation for the methods in the cubic class is found in the remaining ' \
                  f'sections, summarised in the table of contents below.\n\n'

    class_methods = inspect.getmembers(cubic, predicate=inspect.isfunction)
    parent_methods = inspect.getmembers(thermo, predicate=inspect.isfunction)
    cb_methods = sorted(list(set(class_methods) - set(parent_methods)))

    cb_sections = ['Constructor',
                'Utility']

    cb_section_headers = {'Constructor': 'Constructor',
                       'Utility': 'Utility methods'}

    cb_section_intro = {'Constructor': f'Methods to initialise {eosname} model.',
                     'Utility': 'Set- and get methods for interaction parameters, mixing parameters ...'}

    cb_method_dict = split_methods_by_section(cb_sections, cb_methods)
    ofile_text += '## Input keys\n' \
                  '* [Initialiser keys](#initialiser-keys)\n' \
                  '* [Pure fluid &alpha;](#pure-fluid-&alpha;)\n' \
                  '* [&alpha; mixing rules](#&alpha;-mixing-rules)\n' \
                  '* [&beta; mixing rules](#&beta;-mixing-rules)\n\n'
    ofile_text += '# Specific cubics\n\n'
    ofile_text += get_toc(child_sections, child_section_headers, child_method_dict)
    ofile_text += '# Parent class "cubic"\n\n'
    ofile_text += get_toc(cb_sections, cb_section_headers, cb_method_dict)
    ofile_text += intro_text + '\n\n'
    ofile_text += get_markdown_contents(child_sections, child_section_headers, child_section_intro, child_method_dict, sub_toc=False)
    ofile_text += get_markdown_contents(cb_sections, cb_section_headers, cb_section_intro, cb_method_dict)

    filename = f'{classname}_methods.md'
    write_file(MARKDOWN_DIR + filename, ofile_text)



if __name__ == '__main__':
    Functional_to_markdown()
    saft_to_markdown()
    pcsaft_to_markdown()