module GLMakieExt

# Load main package and triggers
using GLMakie
using Statistics
using LogHeightmaps

function init_debug_plots(first_layer, last_layer)
    f = Figure()
    ready = Observable(true)
    exit = Observable(false)
    ax = f[1, 1] = Axis(f, aspect=DataAspect())
    grid = f[2, 1] = GridLayout(tellwidth = false)
    next_button = grid[1,3] = Button(f, label="Next")
    slider = grid[1,2] = Slider(f, range = first_layer:last_layer, startvalue = first_layer)
    prev_button = grid[1,1] = Button(f, label="Prev")
    
    save_button = grid[1,4] = Button(f, label="Save")
    on(slider.value) do _
        ready[] = true
    end
    on(next_button.clicks) do _
        set_close_to!(slider, slider.value[] + 1)
        ready[] = true
    end
    on(prev_button.clicks) do _
        set_close_to!(slider, slider.value[] - 1)
        ready[] = true
    end
    on(events(f.scene).window_open) do event
        if !event
            ready[] = true
            exit[] = true
        end
    end
    on(save_button.clicks) do _
        # CairoMakie.activate!()
        save("figure.png", f)
    end
    display(f)
    function end_action()
        val = slider.value[] + 1 > last_layer ? first_layer : slider.value[] + 1
        set_close_to!(slider, val)
        true
    end
    return slider.value, end_action, exit, ready, ax
end

function do_debug_plots(i, exit, ready, ax, res, dists, circle, points, snake_points, all_snakes, filtered, max_distance)
    println("Layer ", i[])
    println("Residual ", mean(res))
    minD, maxD = extrema(dists)
    szs = 4 .+ (1 .- (dists.-minD)./(maxD.-minD)).^2 .* 2
    empty!(ax)
    GLMakie.lines!(ax, LogHeightmaps.get_circle_points(circle...)..., color=:magenta)
    GLMakie.scatter!(ax, [circle[1]], [circle[2]], markersize=1, color=:magenta)
    # lines!(ax, get_circle_points((circle.+[0,0,max_distance])...)..., color=:magenta, linestyle = :dash)
    # lines!(ax, get_circle_points((circle.+[0,0,-max_distance])...)..., color=:magenta, linestyle = :dash)

    # circle_filtered = (abs.(.√(sum((points[:, 2:3] .- transpose(circle[1:2])).^2, dims=2)) .- circle[3]) .< 10) |> vec
    # println(sum((points[:, 2:3] .- transpose(circle[1:2])).^2, dims=2))
    GLMakie.scatter!(ax, points[:, 2:3] |> transpose, markersize=szs, color=:blue)
    # scatter!(ax, points[circle_filtered, 2:3] |> transpose, markersize=szs[circle_filtered].*0.5, color=:white)
    GLMakie.scatter!(ax, points[filtered, 2:3] |> transpose, markersize=szs[filtered].*0.5, color=:white)
    
    
    GLMakie.lines!(ax, eachcol(snake_points[vcat(1:end, 1), :])..., color=:red)
    GLMakie.lines!(ax, eachcol(shift_snake(snake_points, max_distance)[vcat(1:end, 1), :])..., color=:red, linestyle = :dash)
    GLMakie.lines!(ax, eachcol(shift_snake(snake_points, -max_distance)[vcat(1:end, 1), :])..., color=:red, linestyle = :dash)


    all_dists = LogHeightmaps.distance_to_snake(points[:, 2:3], snake_points)
    snake_dists = vec(minimum(all_dists, dims=1))
    # println(sum(snake_dists .< max_distance))
    GLMakie.scatter!(ax, eachcol(snake_points)..., markersize=10, color=:orange)
    GLMakie.scatter!(ax, eachcol(snake_points[snake_dists .< max_distance, :])..., markersize=8, color=:white)


    # t = range(0, 1; length=size(all_snakes, 1))
    # for i ∈ 1:size(all_snakes, 1)
    #     lines!(ax, eachcol(all_snakes[i, :, :])..., color = get(colorschemes[:magma], t[i], :extrema))
    # end
    ready[] = false
    while !ready[]
        sleep(0.001)
    end
    return exit[]
end

end