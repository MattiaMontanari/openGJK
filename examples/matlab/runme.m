%                           _____      _ _  __                                   %
%                          / ____|    | | |/ /                                   %
%    ___  _ __   ___ _ __ | |  __     | | ' /                                    %
%   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <                                     %
%  | (_) | |_) |  __/ | | | |__| | |__| | . \                                    %
%   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\                                   %
%        | |                                                                     %
%        |_|                                                                     %
%                                                                                %
% Copyright 2022 Mattia Montanari, University of Oxford                          %
%                                                                                %
% This program is free software: you can redistribute it and/or modify it under  %
% the terms of the GNU General Public License as published by the Free Software  %
% Foundation, either version 3 of the License. You should have received a copy   %
% of the GNU General Public License along with this program. If not, visit       %
%                                                                                %
%     https://www.gnu.org/licenses/                                              %
%                                                                                %
% This program is distributed in the hope that it will be useful, but WITHOUT    %
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS  %
% FOR A PARTICULAR PURPOSE. See GNU General Public License for details.          %

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
mex(fullfile('..','..','src','openGJK.c'),...  % Source of openGJK 
    '-largeArrayDims', ...      % Support large arrays
    optflug, ...                % Compiler flag for debug/optimisation
    fullfile('-I..','..','include'),...      % Folder to header files
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