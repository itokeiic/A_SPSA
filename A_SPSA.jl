using Random

function spsa(x0, func; bounds=nothing, alpha=0.602, gamma=0.101, deltax_0=0.1, a=nothing, a_min=1.0e-6, c=1.0e-6, stepredf=0.5, gtol=1.0e-5, graditer=1, memsize=100, IniNfeval=0, maxiter=5000, adaptive_step=true, relaxation=true, dynamic_system=false, args...)
    #INPUT
    #x0: starting input vector, if dynamic_system=true, append 0
    #func: function
    #bounds: [(lower_bound_1, upper_bound_1),(lower_bound_2, upper_bound_2),...,(lower_bound_n, upper_bound_n)], n = # of dim. If nothing, it is automatically set to [(-10,10)]*n
    #alpha: exponential controlling the reduction of step size
    #gamma: exponential controlling the finite difference gradient perturbation magnitude
    #deltax_0: desired minimum initial perturbation of x0
    #stepredf: factor controlling the reduction of stepsize along stochastic gradient descent if no improvement was observed
    #gtol: threshold value below which gradient is considered to be zero
    #graditer: number of times gradients are computed to obtain an averaged stochastic gradient
    #IniNfeval: parameter for accounting for number of function evaluation done before reaching this optimization function
    #maxiter: total number of iteration to be performed
    #adaptive_step: Initial stepsize is automatically reduced if set to true
    redcounter = 0
    if !dynamic_system
        println("Static System")
        Npar = length(x0)
    else
        println("Dynamic System")
        Npar = length(x0) - 1
    end

    function g_sa(x, func, ck, niter, args...) #stochastic gradient calculation
        p = length(x)
        gsum = zeros(p)
        yp = 0.0
        ym = 0.0
        xp = copy(x)
        xm = copy(x)
        delta = zeros(p)
        for m in 1:niter
            delta = 2 .* floor.(2 .* rand(p)) .- 1

            xp = x + ck .* delta
            xm = x - ck .* delta
            if dynamic_system
                xp[end] = xm[end] = x[end]
            end
            yp = func(xp, args...)
            ym = func(xm, args...)
            gsum += (yp - ym) ./ (2 * ck .* delta)
        end
        ghat = gsum / niter
        if dynamic_system
            ghat[end] = 0
        end
        return ghat, yp, ym, xp, xm, delta
    end

    Xmax = Float64[]
    Xmin = Float64[]
    if bounds === nothing
        bounds = [(-10.0, 10.0) for _ in 1:Npar]
        println("No bounds specified. Default:(-10,10).")
    end
    if length(bounds) != Npar
        error("Number of parameters Npar != length of bounds")
    end
    for m in 1:Npar
        push!(Xmin, bounds[m][1])
        push!(Xmax, bounds[m][2])
    end

    Nfeval = IniNfeval
    x0 = collect(x0) # Convert to Vector if it's a tuple or other iterable
    history = []
    historyx = []
    p = length(x0)
    A = Int(floor(0.1 * maxiter))
    y0 = func(x0, args...)
    Nfeval += 1
    mem = fill(y0, memsize)
    x = copy(x0)
    println("initial objective value = ", y0)
    x_best = copy(x0); y_best = y0;
    for k in 1:maxiter
        if dynamic_system
            x[end] = k
        end
        ck = c / (k + 1)^gamma
        ghat, yp, ym, xp, xm, delta = g_sa(x, func, ck, graditer, args...)
        Nfeval += graditer * 2
        if k == 1
            if a === nothing
                a = deltax_0 * (A + 1)^alpha / minimum(abs.(ghat[1:Npar]))
            end
            a_ini = a
            println("ghat0 = ", ghat[:])
        end
        ak = a / (k + 1 + A)^alpha
        println("k: $k, ym = $ym, yp = $yp, a = $a")
        xold = copy(x)
        x = x - ak .* ghat
        for m in 1:Npar
            if x[m] < Xmin[m]
                x[m] = Xmin[m]
            elseif x[m] > Xmax[m]
                x[m] = Xmax[m]
            end
        end
        y = func(x, args...)
        push!(history, [Nfeval, y])
        push!(historyx, copy(x))
        mem = vcat(mem[2:end], min(ym, yp))
        if ym < y_best
            x_best = xm
            y_best = ym
        end
        if yp < y_best
            x_best = xp
            y_best = yp
        end
        if adaptive_step

            if ((y0 - min(yp, ym)) < 0)
                println("divergence detected. reinitializing.")
                redcounter += 1
                x = copy(x_best)
                a = stepredf * a
                if (redcounter > Int(floor(0.05 * maxiter))) && relaxation
                    println("Too many divergence. Resetting a and relaxing threshold!")
                    a = a_ini
                    y0 = min(yp, ym)
                    redcounter = 0
                end
            end
        end
    end
    y = func(x, args...)
    Nfeval += 1
    push!(history, [Nfeval, y])
    push!(historyx, copy(x))
    println("number of function evaluation: ", Nfeval)
    return x, y, history, historyx, Nfeval
end