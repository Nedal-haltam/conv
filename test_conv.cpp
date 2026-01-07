void test_conv2d()
{
    const int w = 5, h = 5;
    int in[h][w];
    int k[KERNEL_HEIGHT][KERNEL_WIDTH];
    // int k[KERNEL_HEIGHT][KERNEL_WIDTH] = {
    //     {0, 1, 0},
    //     {1, 1, 1},
    //     {0, 1, 0},
    // };
    for (int dy = 0; dy < KERNEL_HEIGHT; dy++)
        for (int dx = 0; dx < KERNEL_WIDTH; dx++)
            k[dy][dx] = dy * KERNEL_WIDTH + dx + 1;

    int out[h][w];
    memset(out, 0, sizeof(int) * w * h);
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            in[i][j] = i * w + j + 1;
    // std::cout << "in: \n";
    // for (int i = 0; i < h; i++)
    // {
    //     for (int j = 0; j < w; j++)
    //     std::cout << in[i][j] << " ";
    //     std::cout << "\n";
    // }
    // std::cout << "out: \n";
    // for (int i = 0; i < h; i++)
    // {
    //     for (int j = 0; j < w; j++)
    //         std::cout << out[i][j] << " ";
    //     std::cout << "\n";
    // }


    conv2d(&in[0][0], k, &out[0][0], w, h, 0, 0, 1, 0);

    // std::cout << "in: \n";
    // for (int i = 0; i < 5; i++)
    // {
    //     for (int j = 0; j < 5; j++)
    //     std::cout << in[i][j] << " ";
    //     std::cout << "\n";
    // }
    // std::cout << "out: \n";
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
            std::cout << out[i][j] << "\n";
            // std::cout << out[i][j] << " ";
        // std::cout << "\n";
    }
}

void test_conv3d()
{
    const int w = 5, h = 5, d = 5;
    int in[d][h][w];
    int k[KERNEL_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    // int k[KERNEL_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH] = {
    //     {
    //         {0, 0, 0},
    //         {0, 0, 0},
    //         {0, 0, 0},
    //     },
    //     {
    //         {0, 0, 0},
    //         {0, 1, 0},
    //         {0, 0, 0},
    //     },
    //     {
    //         {0, 0, 0},
    //         {0, 0, 0},
    //         {0, 0, 0},
    //     },
    // };
    for (int dz = 0; dz < KERNEL_DEPTH; dz++)
        for (int dy = 0; dy < KERNEL_HEIGHT; dy++)
            for (int dx = 0; dx < KERNEL_WIDTH; dx++)
                k[dz][dy][dx] = dz * KERNEL_WIDTH * KERNEL_HEIGHT + dy * KERNEL_WIDTH + dx + 1;

    int out[d][h][w];
    memset(out, 0, sizeof(int) * w * h * d);
    for (int z = 0; z < d; z++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                in[z][y][x] = z * w * h + y * w + x + 1;

    // std::cout << "in: \n";
    // for (int z = 0; z < d; z++) {
    //     for (int y = 0; y < h; y++) {
    //         for (int x = 0; x < w; x++) {
    //             std::cout << in[z][y][x] << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n---------------------------------\n";
    // }
    // std::cout << "out: \n";
    // for (int z = 0; z < d; z++) {
    //     for (int y = 0; y < h; y++) {
    //         for (int x = 0; x < w; x++) {
    //             std::cout << out[z][y][x] << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n---------------------------------\n";
    // }

    conv3d(&in[0][0][0], k, &out[0][0][0], w, h, d, 0, 0, 0, 1, 0);

    // std::cout << "out: \n";
    for (int z = 0; z < d; z++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // std::cout << out[z][y][x] << " ";
                std::cout << out[z][y][x] << "\n";
            }
            // std::cout << "\n";
        }
        // std::cout << "\n---------------------------------\n";
    }
}
