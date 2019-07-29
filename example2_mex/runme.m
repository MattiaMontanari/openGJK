% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %
%                                    #####        # #    #                %
%        ####  #####  ###### #    # #     #       # #   #                 %
%       #    # #    # #      ##   # #             # #  #                  %
%       #    # #    # #####  # #  # #  ####       # ###                   %
%       #    # #####  #      #  # # #     # #     # #  #                  %
%       #    # #      #      #   ## #     # #     # #   #                 %
%        ####  #      ###### #    #  #####   #####  #    #                %
%                                                                         %
%   This file is part of openGJK.                                         %
%                                                                         %
%   openGJK is free software: you can redistribute it and/or modify       %
%    it under the terms of the GNU General Public License as published by %
%    the Free Software Foundation, either version 3 of the License, or    %
%    any later version.                                                   %
%                                                                         %
%    openGJK is distributed in the hope that it will be useful,           %
%    but WITHOUT ANY WARRANTY; without even the implied warranty of       %
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See The        %
%    GNU General Public License for more details.                         %
%                                                                         %
%   You should have received a copy of the GNU General Public License     %
%    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.     %
%                                                                         %
%        openGJK: open-source Gilbert-Johnson-Keerthi algorithm           %
%             Copyright (C) Mattia Montanari 2018 - 2019                  %
%               http://iel.eng.ox.ac.uk/?page_id=504                      %
%                                                                         %
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %
%                                                                         %
% This file compiles a mex function from the openGJK library and runs an  %
%   example. If the mex function cannot be compiled an error is returned. %
%                                                                         %
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %

% CLEAR ALL VARIABLES
clearvars

% SELECT OPTIMISATION FLAG - FASTER BUT NOT SUITABLE FOR DEBUGGING
if 0
    optflug = '-g'; %#ok<*UNRCH>
else
    optflug = '-O';
end
% SELECT SILET COMPILATION MODE.
if 1 
    silflag = '-silent'; 
else
    silflag = '-v';
end

% TRY COMPILING MEX FILE
fprintf('Compiling mex function... ')
try
mex(fullfile('..','lib','src','openGJK.c'),...  % Source of openGJK 
    '-largeArrayDims', ...      % Support large arrays
    optflug, ...                % Compiler flag for debug/optimisation
    fullfile('-I..','lib','include'),...      % Folder to header files
    '-outdir', pwd,...          % Ouput directory for writing mex function
    '-output', 'openGJK',...    % Name of ouput mex file
    '-DMATLABDOESMEXSTUFF',...  % Define variable for mex function in source files
    silflag )                   % Silent/verbose flag

    % File compiled without errors. Return path and name of mex file
    fprintf('completed!\n')
    fprintf('The following mex file has been generated:')
    fprintf('\t%s\n',[pwd,filesep,'openGJK.',mexext]) 
catch
    % Build failed, refer to documentation
    fprintf('\n\n ERROR DETECTED! Mex file cannot be compiled.\n')
    fprintf('\tFor more information, see ')
    fprintf('<a href="http://www.mathworks.com/help/matlab/ref/mex.html">this documentation page</a>.\n\n')
    return
end
 
% RUN EXAMPLE
fprintf('Running example... ')
main
fprintf('completed!\n')