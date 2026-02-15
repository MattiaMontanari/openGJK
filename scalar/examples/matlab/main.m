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

% DEFINE BODY A AS 3xN MATRIX, WHERE N IS THE NUMBER OF VERTICES OF BODY A
A = [  0.0  2.3  8.1  4.3  2.5  7.1  1.0  3.3  6.0
       5.5  1.0  4.0  5.0  1.0  1.0  1.5  0.5  1.4
       0.0 -2.0  2.4  2.2  2.3  2.4  0.3  0.3  0.2];

% DEFINE BODY B IN THE OPPOSITE QUADRANT OF BODY A
B = -A; 

% COMPUTE MINIMUM DISTANCE AND RETURN VALUE
dist = openGJK( A, B ); 
fprintf('The minimum distance between A and B is %.2f\n',dist);

% VISUALISE RESULTS ONLY IN MATLAB
if(exist('OCTAVE_VERSION', 'builtin') == 0)
       % .. create new figure
       figure('units','centimeters', 'WindowStyle','normal', 'color','w',...
       'Position',[0 8.5 9 6],'defaultAxesColorOrder',parula,...
       'Renderer','opengl') 
       % .. adjust properties
       axis equal tight off; hold all; 
       % .. display body A
       DT = delaunayTriangulation(A');
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .9 ],...
              'FaceLighting','flat' )
       % .. display body B
       DT = delaunayTriangulation(B');
       [K,~] = convexHull(DT);
       trisurf(K,DT.Points(:,1),DT.Points(:,2),DT.Points(:,3),...
              'EdgeColor','none','FaceColor',[.4 1 .8 ],...
              'FaceLighting','flat' )
       % .. represent the computed distance as a sphere
       [x,y,z] = sphere(100);
       surf(x.*dist/2,y.*dist/2,z.*dist/2,'facecolor',[.9 .9 .9],...
       'EdgeColor','none','FaceLighting','flat','SpecularColorReflectance',0,...
       'SpecularStrength',1,'SpecularExponent',10,'facealpha',.7)
       % ... adjust point of view   
       view(42,21)
       % ... add light
       light('Position',[5 -10 20],'Style','local'); 
end