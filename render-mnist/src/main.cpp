#include <iostream>
using namespace std;

#include <vector>
#include <string>
#include <fstream>

#include "renderclass.h"

void getFiles(string parentFolder, vector<string> & vecFileNames) {

	string cmd = "ls " + parentFolder + "/*.png > temp.log";
	system(cmd.c_str());

	ifstream ifs("temp.log");
	//������log���������������������
	if (ifs.fail()) {
		return;
	}

	string fileName;
	while (getline(ifs, fileName)) {
		vecFileNames.push_back(fileName);
	}

	ifs.close();
	return;
}

void createplaneobj() {

	FILE *fp = fopen("obj.obj", "w");
	fprintf(fp, "mtllib model_normalized.mtl\n");

	int ynum = 2;
	int xnum = 2;
	for (int y = 0; y < ynum; y++)
		for (int x = 0; x < xnum; x++) {
			{

				float xx = 1.0f * x / (xnum - 1) * 2 - 1;
				float yy = 1.0f * y / (ynum - 1) * 2 - 1;
				float zz = 0.0f;

				float rtio = 0.45;
				fprintf(fp, "v %.3f %.3f %.3f\n", xx * rtio, yy * rtio,
						zz * rtio);
			}
		}

	for (int y = 0; y < ynum; y++)
		for (int x = 0; x < xnum; x++) {
			{

				float xx = 1.0f * x / (xnum - 1);
				float yy = 1.0f * y / (ynum - 1);

				fprintf(fp, "vt %.3f %.3f\n", xx, yy);
			}
		}
	fprintf(fp, "usemtl material_0_1_8\n");

	for (int y = 0; y < ynum - 1; y++)
		for (int x = 0; x < xnum - 1; x++) {
			{
				int pidx = y * xnum + x + 1;
				int pright = pidx + 1;
				int pdown = pidx + ynum;
				int pdownright = pdown + 1;

				fprintf(fp, "f %d/%d %d/%d %d/%d\n", pidx, pidx, pdownright,
						pdownright, pdown, pdown);
				fprintf(fp, "f %d/%d %d/%d %d/%d\n", pidx, pidx, pright, pright,
						pdownright, pdownright);
			}
		}

	fclose(fp);
}

int main() {

	createplaneobj();

	string parentFlder = "mnist dataset which contains 60000 mnist images";

	// vector<string> folders;
	// getFiles(parentFlder, folders);

	string parentSvFolder = "your save folder";
	for (int i = 0; i < 1; i++) {
		string svfolder = parentSvFolder + "/" + to_string(i);

		char cmd[256];
		sprintf(cmd, "mkdir %s", svfolder.c_str());
		system(cmd);
	}

	int height = 256;
	int width = 256;

	int maxsz = 50;

	render *tmp = new render(height, width, maxsz);
	tmp->initializecuda();
	tmp->programobj();

	int stepbe = 0;
	int stepen = 60000;
	int step = 0;

	for (int i = 0; i< 10; i++) {
		char imidx[50];
		sprintf(imidx, "im%05d.png", i);
		string imname = parentFlder + "/" + string(imidx);
		cout << "imname" << "\t" << imname << endl;
		cv::Mat im = cv::imread(imname);
		cv::imwrite("./re.png", im);
		string name = "obj.obj";

		bool suc = true;
		mesh tmpobj = tmp->loadobj("./", name, suc);
		if (!suc) {
			continue;
		} else {

			step++;
			if (!(step >= stepbe && step < stepen))
				continue;

			tmp->loadmesh(tmpobj);

			int rnum = 2;
			int lvnum = 7;
			int lhnum = 7;
			int shininesslevel = 0;
			for (shininesslevel = 0; shininesslevel < 1; shininesslevel++) {

				int pos = imname.find_last_of('/');
				string svfolder = parentSvFolder + "/"
						+ to_string(shininesslevel) + "/"
						+ imname.substr(pos + 1,
								imname.length() - pos - 5);
				cout << imname << "\t" << name << endl;
				cout << "svfolder\t" << svfolder << endl;

				char cmd[256];
				sprintf(cmd, "mkdir %s", svfolder.c_str());
				system(cmd);

				int sz = maxsz;
				int hnum = maxsz * (2 + shininesslevel);
				int vnum = maxsz * (2 + shininesslevel);

				tmp->display(svfolder, tmpobj, shininesslevel, sz, rnum, lhnum,
						lvnum, hnum, vnum);
			}

			tmp->deletemesh();
		}
	}

	delete tmp;

	cout << "done!" << endl;
	return 0;
}

