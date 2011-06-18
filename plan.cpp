#include "plan.h"
#include <ini/minIni.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>

using namespace std;

static void print_usage_exit(const char * program_file)
{
	fprintf(stderr, "Usage: %s ini_file\n", program_file);
	exit(31);
}

static bool exists(const char *filename)
{
	ifstream ifile(filename);
	return ifile;
}

static void output_file_check(const char * fn)
{
	if (exists(fn))
	{
		fprintf(stderr, "Output file %s is already exists. Overwrite? [y/n]: ", fn);
		char res;
		cin >> res;
		if (res == 'y')
			return;
		fprintf(stderr, "Exit\n");
		exit(33);
	}
} 

Plan::Plan(int argc, char ** argv)
{
	if (argc == 1 || !exists(argv[1]))
		print_usage_exit(argv[0]);
	minIni ini(argv[1]);
	string max_fn = ini.gets("General", "MaxFileName");
	string res_fn = ini.gets("General", "ResFileName");
	if (max_fn == "" || res_fn == "")
	{
		fprintf(stderr, "Bad ini\n");
		exit(32);
	}
	pref_gpu = (int)ini.getl("General", "GPU", 0);
	load_max = ini.getbool("General", "LoadMax", false);
	sz = ini.getf("General", "Size", -1);
	if (sz <= 0)
	{
		fprintf(stderr, "Bad size parameter\n");
		exit(34);
	}
	if (!load_max)
	{
		R = ini.getf("General", "R", -1);
		if (R <= 0)
                {
                        fprintf(stderr, "Bad radius parameter\n");
                        exit(35);
	        }
		Emaxpack = ini.getf("General", "Emaxpack");
		if (Emaxpack <= 0 || Emaxpack >= 1)
                {
                        fprintf(stderr, "Bad Emaxpack parameter\n");
                        exit(35);
	        }
	}
	Eres = ini.getf("General", "Eres", -1);
	if (Eres <= 0 || Eres >= 1)
	{
		fprintf(stderr, "Bad Eres parameter\n");
		exit(36);
	}
	
	if (!load_max)
		output_file_check(max_fn.c_str());
	output_file_check(res_fn.c_str());

	max_file_name = new char[max_fn.size() + 1];
	strcpy(max_file_name, max_fn.c_str());

	res_file_name = new char[res_fn.size() + 1];
	strcpy(res_file_name, res_fn.c_str());
}

Plan::~Plan()
{
	delete [] max_file_name;
	delete [] res_file_name;
}
