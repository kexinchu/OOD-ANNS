#include "data_loader.h"
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_bin_path> <output_bin_path> <ratio (0.0-1.0)>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    double ratio = std::stod(argv[3]);

    if (ratio <= 0.0 || ratio > 1.0) {
        std::cerr << "Error: ratio must be between 0.0 and 1.0\n";
        return 1;
    }

    // Read input file header - ensure file is properly opened fresh
    std::ifstream fin;
    fin.open(input_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open input file " << input_path << "\n";
        return 1;
    }

    // Ensure we're at the beginning of the file
    fin.seekg(0, std::ios::beg);
    
    size_t n, d;
    fin.read((char*)&n, 4);
    if (fin.gcount() != 4) {
        std::cerr << "Error: Failed to read n from file header\n";
        fin.close();
        return 1;
    }
    fin.read((char*)&d, 4);
    if (fin.gcount() != 4) {
        std::cerr << "Error: Failed to read d from file header\n";
        fin.close();
        return 1;
    }

    std::cout << "Input file: " << input_path << "\n";
    std::cout << "Total number: " << n << ", dimension: " << d << "\n";

    size_t subset_n = (size_t)(n * ratio);
    std::cout << "Extracting " << subset_n << " vectors (" << (ratio * 100) << "%)\n";

    // Write output file header
    std::ofstream fout;
    fout.open(output_path, std::ios::out | std::ios::binary);
    if (!fout.is_open()) {
        std::cerr << "Error: Cannot create output file " << output_path << "\n";
        fin.close();
        return 1;
    }

    fout.write((char*)&subset_n, 4);
    fout.write((char*)&d, 4);

    // Copy data - use generic buffer to support both int and float
    char* buffer = new char[d * sizeof(int)];  // Use sizeof(int) as max size (4 bytes)
    size_t element_size = sizeof(int);  // Default to int size, but we'll detect from file size
    
    // Try to detect element size by checking if we can read first element
    // For simplicity, we'll assume files are either int (4 bytes) or float (4 bytes)
    // Both are same size, so we can use a generic approach
    size_t bytes_per_vector = d * 4;  // Assume 4 bytes per element (int or float)
    
    for (size_t i = 0; i < subset_n; ++i) {
        fin.read(buffer, bytes_per_vector);
        size_t bytes_read = fin.gcount();
        if (bytes_read != bytes_per_vector) {
            std::cerr << "Error: Failed to read complete vector at index " << i << "\n";
            delete[] buffer;
            fin.close();
            fout.close();
            return 1;
        }
        fout.write(buffer, bytes_per_vector);
    }

    delete[] buffer;
    fin.close();
    fout.close();

    std::cout << "Successfully extracted to " << output_path << "\n";
    return 0;
}

