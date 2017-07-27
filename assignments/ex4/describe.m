% https://stackoverflow.com/a/45347880/5353461

function varargout = describe(varargin)
    % varargin used to accomodate variable number of input names
    st = dbstack;
    outstring = '';
    for ii = size(st, 1):-1:2
        outstring = [outstring, st(ii).file, ' > ', st(ii).name, ', line ', num2str(st(ii).line), '\n'];
    end
    % Loop over variables and get info
    for n = 1:nargin
        % Variables are passed by name, so check if they exist
        try v = evalin('caller', varargin{n});
            outstring = [outstring, '"', varargin{n}, '" is a ', typeinfo(v), ' of size ', mat2str(size(v)), '\n'];   
        catch
            outstring = [outstring, 'Variable "', varargin{n}, '" not found!\n'];
        end
    end
    fprintf(outstring)
end
