% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %
%                                    #####        # #    #                %
%        ####  #####  ###### #    # #     #       # #   #                 %
%       #    # #    # #      ##   # #             # #  #                  %
%       #    # #    # #####  # #  # #  ####       # ###                   %
%       #    # #####  #      #  # # #     # #     # #  #                  %
%       #    # #      #      #   ## #     # #     # #   #                 %
%        ####  #      ###### #    #  #####   #####  #    #                %
%                                                                         %
%           Mattia Montanari    |   University of Oxford 2018             %
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %
%                                                                         %
% This file runs an example to illustrate how to invoke the openGJK lib   %
%   withing Matlab. It that assumes a mex file openGJK is availalbe, see  %
%   the runme.m script for information on how to compile it.              % 
% The example computes the minimum distance between two polytopes in 3D,  %
%   A and B, both defined as a list of points.                            %
%                                                                         %
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %

% DEFINE BODY A AS 3xN MATRIX, WHERE N IS THE NUMBER OF VERTICES OF BODY A
A = [  0.0  2.3  8.1  4.3  2.5  7.1  1.0  3.3  6.0
       5.5  1.0  4.0  5.0  1.0  1.0  1.5  0.5  1.4
       0.0 -2.0  2.4  2.2  2.3  2.4  0.3  0.3  0.2];

% DEFINE BODY B IN THE OPPOSITE QUADRANT OF BODY A
B = -A; 

% COMPUTE MINIMUM DISTANCE AND RETURN VALUE
dist = openGJK( A, B ); 
fprintf('The minimum distance between A and B is %.2f\n',dist);

% VISUALISE RESULTS
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
