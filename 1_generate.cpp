#include "Halide.h"
#include "halide_image_io.h"
using namespace Halide::Tools;
#include<iostream>

namespace {

using namespace Halide;

    class MyFirstGenerator : public Halide::Generator<MyFirstGenerator>{
        public:
            Input<Buffer<uint8_t>>  input{"futurama", 2};
            Output<Buffer<uint8_t>>  output{"lesson_2_futurama", 2};
            
            void generate(){
                var x("x"), y("y"), c("c");
                Func hw_input("hw_input");
                Func brighter;
                hw_input = cast<int>(input(x,y,c));
                

                //value = cast<int>(input(x,y,c));
                value = hw_input + 50;
                value = min(value, 255.0f);
                value = cast<uint8_t>(value);
                brighter(x,y,c) = value;

                func hw_output("hw_output");
                hw_output(x, y, c) = brighter(x, y, c);
                output(x,y,c) = cast<uint8_t>(hw_output(x,y,c));


                /* THE SCHEDULE */
                if (get_target().has_feature(Target::CoreIR)) {
                Var xi,yi, xo,yo;
                
                hw_input.compute_root();
                hw_output.compute_root();

                output.bound(x, 0, input.width-1);
                output.bound(y, 0, output.width-1);
                
                hw_output.tile(x,y, xo,yo, xi,yi, input.width()-1, input.height()-1)
                    .hw_accelerate(xi, xo);

                brighter.update()
                    .unroll(r.x, 2)
                    .unroll(r.y, 2);

                brighter.linebuffer();

                hw_input.compute_at(hw_output, xi).store_at(hw_output, xo);
                hw_input.stream_to_accelerator();

                } else if (get_target().has_feature   (Target::Clockwork)) {
                Var xi,yi, xo,yo;

                output.bound(x, 0, input.width-1);
                output.bound(y, 0, output.width-1);
                
                hw_output.compute_root();

                hw_output.tile(x,y, xo,yo, xi,yi, input.width()-1, input.height()-1)
                    .hw_accelerate(xi, xo);
                
                
                brighter.compute_at(hw_output, xo);
                brighter.update()
                    .unroll(r.x, 2)
                    .unroll(r.y, 2);
                

                hw_input.stream_to_accelerator();

                } else {  // schedule to CPU
                brighter.compute_root();
                brighter.update()
                    .unroll(r.x, 2)
                    .unroll(r.y, 2);
                }
            }
    };

}  // namespace

HALIDE_REGISTER_GENERATOR(MyFirstGenerator, 1_generate)


