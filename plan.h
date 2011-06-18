#ifndef _PLAN_H_

#define _PLAN_H_

class Plan
{
public:
	Plan(int argc, char ** argv);
	~Plan();
	bool load_max;
	int pref_gpu;
	char * max_file_name;
	char * res_file_name;
	float sz;
	float R;
	float Emaxpack;
	float Eres;
};


#endif
