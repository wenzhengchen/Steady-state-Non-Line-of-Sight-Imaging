#include "renderclass.h"

////////////////////////////////////////////////////////////////////////////////

inline float randomnum() {
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

inline void saveim(vector<float> tmpdata, cv::Mat tmpshow, cv::Mat tmpsave,
		int height, int width, char *fold, char *fname, int id) {

	//after drawing
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, width, height, GL_RGB,
	GL_FLOAT, &tmpdata[0]);

	float maxval = 0.0f;
	for (int ih = 0; ih < height; ih++)
		for (int jw = 0; jw < width; jw++)
			for (int kc = 0; kc < 3; kc++) {
				int idx = (height - 1 - ih) * width * 3 + jw * 3 + 2 - kc;

				tmpsave.at<cv::Vec3f>(ih, jw)[kc] = tmpdata[idx];

				if (tmpdata[idx] > maxval)
					maxval = tmpdata[idx];
			}

	for (int ih = 0; ih < height; ih++)
		for (int jw = 0; jw < width; jw++)
			for (int kc = 0; kc < 3; kc++) {
				int idx = (height - 1 - ih) * width * 3 + jw * 3 + 2 - kc;

				tmpshow.at<cv::Vec3b>(ih, jw)[kc] = (uchar) (255.0f
						* tmpdata[idx] / maxval);
			}

	// cv::imshow("tmp", tmpshow);
	// cv::waitKey(0);

	char name[256];
	sprintf(name, "%s/%s-%d-%.4f.png", fold, fname, id, maxval);
	cv::imwrite(name, tmpshow);

	// Write to file!
	sprintf(name, "%s/%s-%d-%.4f.txt", fold, fname, id, maxval);
	cv::FileStorage fs(name, cv::FileStorage::WRITE);
	fs << "mat1" << tmpsave;

	return;
}

inline void savedep(vector<float> tmpdata, cv::Mat tmpshow, cv::Mat tmpsave,
		int height, int width, char *fold, char *fname, int id) {

	//after drawing
	//after drawing
	glReadBuffer(GL_DEPTH_ATTACHMENT);
	glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT,
	GL_FLOAT, &tmpdata[0]);

	for (int ih = 0; ih < height; ih++)
		for (int jw = 0; jw < width; jw++) {

			int idx = (height - 1 - ih) * width + jw;

			float dis = tmpdata[idx];
			uchar dis255;

			// background
			if (abs(dis - 1) < 1e-5) {
				dis = -1;
				dis255 = 0;

			} else {
				dis = dis * 2 - 1;
				dis = -dis;
				dis = dis * 100;
				dis255 = (uchar) 255.0f * (dis + 1) / 2;

				if (dis > 1)
					dis255 = 255;
				if (dis < -1)
					dis255 = 0;
			}

			tmpshow.at<cv::Vec3b>(ih, jw) = cv::Vec3b(dis255, dis255, dis255);
			tmpsave.at<cv::Vec3f>(ih, jw) = cv::Vec3f(dis, dis, dis);

		}

	// cv::imshow("tmp", tmpshow);
	// cv::waitKey(0);

	char name[256];
	sprintf(name, "%s/%s-%d.png", fold, fname, id);
	cv::imwrite(name, tmpshow);

	// Write to file!
	sprintf(name, "%s/%s-%d.txt", fold, fname, id);
	cv::FileStorage fs(name, cv::FileStorage::WRITE);
	fs << "mat1" << tmpsave;

	return;
}

void render::display(string svfolder, mesh obj, int shininesslevel, int sz,
		int rnum, int lighthnum, int lightvnum, int hnum, int vnum) {

	hnum = SAMRATE * hnum / 10;
	vnum = SAMRATE * vnum / 10;
	sz = SAMRATE * sz / 10;

	if (sz > maxsz) {
		cout << "maxsz\t" << maxsz << endl;
		cout << "sz\t" << sz << endl;
		cout << "sz cannot be larger than maxsz" << endl;
		sz = maxsz;
	}

	int texblockdim = hnum / sz;
	int texthreaddim = sz;
	cout << "tblock dim\t" << texblockdim << endl;
	cout << "tthread dim\t" << texthreaddim << endl;

	////////////////////////////////////////////////////
	std::vector<float> tmpdata(width * height * 3);
	cv::Mat tmpshow = cv::Mat::zeros(height, width, CV_8UC3);
	cv::Mat tmpsave = cv::Mat::zeros(height, width, CV_32FC3);

	glm::mat4 I = glm::mat4(1.0f);
	vector<glm::mat4> views = getViewMatrix(hnum * vnum);

	//////////////////////////////////////////////////////////////////
	glBindFramebuffer(GL_FRAMEBUFFER, fbo_obj);
	glUseProgram(programID);

	/////////////////////////////////////////////////////////////////////
	// 3 levels
	// 1 is low specular, 0-1
	// 2 is midium, 1-256
	// 3 is high, 256-512

	// how many shininess we sample
	// how many rotations translations we have
	for (int rid = 0; rid < rnum; rid++) {

		float r = randomnum();

		float shiness = -1;
		switch (shininesslevel) {
		case 0:
			shiness = 0.0f;
			break;
		case 1:
			shiness = 63.0f * r + 1.0f;
			break;
		case 2:
			shiness = 64.0f + 192.0f * r;
			break;
		case 3:
			shiness = 256.0f + 256.0f * r;
			break;
		default:
			cout << "unknow specular" << endl;
		}

		std::cout << "shininess\t" << shiness << std::endl;

		/*
		 float xrot = atof(rots[0].c_str());
		 float yrot = atof(rots[1].c_str());
		 float zrot = atof(rots[2].c_str());

		 // x, y move [-0.3, 0.3]
		 // z move [0, 0.8]
		 float xshift = atof(shifts[0].c_str());
		 float yshift = atof(shifts[1].c_str());
		 float zshift = atof(shifts[2].c_str());
		 */

		float xrot = 30 * (2 * randomnum() - 1);
		float yrot = 30 * (2 * randomnum() - 1);
		float zrot = 180 * (2 * randomnum() - 1);

		// x, y move [-0.4, 0.4]
		// z move [-0.2, 0.8]
		float xshift = 0.6f * (2 * randomnum() - 1);
		float yshift = 0.6f * (2 * randomnum() - 1);
		float zshift = 0.2 + 0.6f * (2 * randomnum() - 1);

		glm::mat4 ModelMatrix = getModelMatrix(xrot, yrot, zrot, xshift, yshift,
				zshift);

		char folder[256];
		sprintf(folder, "%s/shine_%.4f-rot_%.4f_%.4f_%.4f-shift_%.4f_%.4f_%.4f",
				svfolder.c_str(), shiness, xrot, yrot, zrot, xshift, yshift,
				zshift);

		char cmd[256];
		sprintf(cmd, "mkdir %s", folder);
		system(cmd);

		//////////////////////////////////////////////////////////
		glBindFramebuffer(GL_FRAMEBUFFER, fbo_obj);
		glUseProgram(programID);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDisable(GL_BLEND);

		glUniformMatrix4fv(ModelMatrixID, 1,
		GL_FALSE, &ModelMatrix[0][0]);

		//////////////////////////////////////////////////////////
		// draw ortho image
		glUniformMatrix4fv(ViewMatrixID, 1,
		GL_FALSE, &I[0][0]);

		glUniform3f(lightPosition_modelspace, 0.0f, 0.0f, 1.0f);

		// Clear the screen
		glClear(
		GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for (int i = 0; i < objnums; i++) {

			glBindVertexArray(VAOs[i]);

			// draw an original image
			glUniform1f(Shininess, 0.0f);

			// glUniform3f(MaterialAmbient, 0.1f, 0.1f, 0.1f);
			glUniform3f(MaterialAmbient, obj.mas[i].Ka[0], obj.mas[i].Ka[1],
					obj.mas[i].Ka[2]);
			// glUniform3f(MaterialDiffuse, 1.0f, 1.0f, 1.0f);
			glUniform3f(MaterialDiffuse, obj.mas[i].Kd[0], obj.mas[i].Kd[1],
					obj.mas[i].Kd[2]);
			glUniform3f(MaterialSpecular, obj.mas[i].Ks[0], obj.mas[i].Ks[1],
					obj.mas[i].Ks[2]);

			if (obj.meshes[i].istex) {
				// Bind our texture in Texture Unit 0
				glActiveTexture(GL_TEXTURE0 + tid + i);
				glBindTexture(GL_TEXTURE_2D, textures[i]);

				// Set our "myTextureSampler" sampler to use Texture Unit 0
				glUniform1i(TextureID, tid + i);
				glUniform1f(istex, 1.0f);
			} else {
				glUniform1f(istex, -1.0f);
			}

			// Draw the triangles !
			glDrawArrays(GL_TRIANGLES, 0, obj.meshes[i].vertices.size());
		}

		/*
		 // Bind our texture in Texture Unit 1
		 glActiveTexture(GL_TEXTURE0 + tid - 1);
		 glBindTexture(GL_TEXTURE_2D, tex_saver);
		 glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
		 */

		saveim(tmpdata, tmpshow, tmpsave, height, width, folder, "original", 0);
		savedep(tmpdata, tmpshow, tmpsave, height, width, folder, "depth", 0);

		///////////////////////////////////////////////////////////////////////
		/*
		 glBindFramebuffer(GL_FRAMEBUFFER, fbo_acc);
		 glUseProgram(programQuad);

		 glClear(
		 GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		 glDisable(GL_DEPTH_TEST);
		 glEnable(GL_BLEND);
		 glBlendFunc(GL_SRC_ALPHA,
		 GL_ONE_MINUS_SRC_ALPHA);
		 glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO,
		 GL_ONE);

		 glBindVertexArray(quadVAO);

		 glActiveTexture(GL_TEXTURE0 + tid - 1);
		 glBindTexture(GL_TEXTURE_2D, tex_saver);

		 glUniform1i(TextureIdx, 0);

		 // Draw the triangles !
		 glDrawArrays(GL_TRIANGLES, 0, 6);

		 saveim(tmpdata, tmpshow, tmpsave, height, width, folder, "original", 1);
		 savedep(tmpdata, tmpshow, tmpsave, height, width, folder, "depth", 1);
		 */

		///////////////////////////////////////////////////////////////
		// draw combined image
		float maxvalue = 0;
		vector<cv::Mat> tmpaccs;
		for (int lh = 0; lh < lighthnum; lh++) {
			for (int lv = 0; lv < lightvnum; lv++) {

				bool hbor = (lh == 0) | (lh == lighthnum - 1);
				bool vbor = (lv == 0) | (lv == lightvnum - 1);
				if (!(hbor | vbor))
					continue;

				cout << "light idx" << lh * lightvnum + lv << endl;
				// float light_x = 2.0f * (lh + 1) / (lighthnum + 1) - 1.0f;
				// float light_y = 2.0f * (lv + 1) / (lightvnum + 1) - 1.0f;

				float light_x = 0.4f * (lh - 3);
				float light_y = 0.4f * (lv - 3);
				float light_z = 1.0f;

				glBindFramebuffer(GL_FRAMEBUFFER, fbo_obj);
				glUseProgram(programID);
				glUniform3f(lightPosition_modelspace, light_x, light_y,
						light_z);

				glBindFramebuffer(GL_FRAMEBUFFER, fbo_acc);
				glClear(
				GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				for (int hb = 0; hb < texblockdim; hb++) {
					for (int wb = 0; wb < texblockdim; wb++) {

						glBindFramebuffer(GL_FRAMEBUFFER, fbo_obj);
						glUseProgram(programID);

						glEnable(GL_DEPTH_TEST);
						glDepthFunc(GL_LESS);
						glDisable(GL_BLEND);

						for (int ht = 0; ht < texthreaddim; ht++) {
							for (int wt = 0; wt < texthreaddim; wt++) {

								int xshift = wt * width;
								int yshift = ht * height;

								int renderid = hb * texblockdim * texthreaddim
										* texthreaddim
										+ wb * texthreaddim * texthreaddim
										+ ht * texthreaddim + wt;

								glUniformMatrix4fv(ViewMatrixID, 1,
								GL_FALSE, &views[renderid][0][0]);

								// Clear the screen
								glClear(
								GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

								for (int i = 0; i < objnums; i++) {

									glBindVertexArray(VAOs[i]);

									// draw an original image
									glUniform1f(Shininess, shiness);

									// glUniform3f(MaterialAmbient, 0.1f, 0.1f, 0.1f);
									glUniform3f(MaterialAmbient,
											obj.mas[i].Ka[0], obj.mas[i].Ka[1],
											obj.mas[i].Ka[2]);
									// glUniform3f(MaterialDiffuse, 1.0f, 1.0f, 1.0f);
									glUniform3f(MaterialDiffuse,
											obj.mas[i].Kd[0], obj.mas[i].Kd[1],
											obj.mas[i].Kd[2]);
									glUniform3f(MaterialSpecular,
											obj.mas[i].Ks[0], obj.mas[i].Ks[1],
											obj.mas[i].Ks[2]);

									if (obj.meshes[i].istex) {
										// Bind our texture in Texture Unit 0
										glActiveTexture(
										GL_TEXTURE0 + tid + i);
										glBindTexture(GL_TEXTURE_2D,
												textures[i]);

										// Set our "myTextureSampler" sampler to use Texture Unit 0
										glUniform1i(TextureID, tid + i);
										glUniform1f(istex, 1.0f);
									} else {
										glUniform1f(istex, -1.0f);
									}

									// Draw the triangles !
									glDrawArrays(GL_TRIANGLES, 0,
											obj.meshes[i].vertices.size());
								}

								// Bind our texture in Texture Unit 1
								glActiveTexture(GL_TEXTURE0 + tid - 1);
								glBindTexture(GL_TEXTURE_2D, tex_saver);
								glCopyTexSubImage2D(GL_TEXTURE_2D, 0, xshift,
										yshift, 0, 0, width, height);

								//saveim(tmpdata, tmpshow, height, width, folder,
								//	"single", renderid);
							}
						}

						glBindFramebuffer(GL_FRAMEBUFFER, fbo_acc);
						glUseProgram(programQuad);

						glDisable(GL_DEPTH_TEST);
						glEnable(GL_BLEND);
						glBlendFunc(GL_SRC_ALPHA,
						GL_ONE_MINUS_SRC_ALPHA);
						glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO,
						GL_ONE);

						glBindVertexArray(quadVAO);

						glActiveTexture(GL_TEXTURE0 + tid - 1);
						glBindTexture(GL_TEXTURE_2D, tex_saver);

						for (int ht = 0; ht < texthreaddim; ht++) {
							for (int wt = 0; wt < texthreaddim; wt++) {

								glUniform1i(TextureIdx, ht * maxsz + wt);

								// bend

								// Draw the triangles !
								glDrawArrays(GL_TRIANGLES, 0, 6);
								// saveim(tmpdata, tmpshow, height, width, folder, "blend",
								//	renderid);
							}
						}

					}
				}

				//after drawing
				glReadBuffer(GL_COLOR_ATTACHMENT0);
				glReadPixels(0, 0, width, height, GL_RGB,
				GL_FLOAT, &tmpdata[0]);

				cv::Mat tmpacc = cv::Mat::zeros(height, width,
				CV_32FC3);
				for (int ih = 0; ih < height; ih++)
					for (int jw = 0; jw < width; jw++)
						for (int kc = 0; kc < 3; kc++) {
							int idx = (height - 1 - ih) * width * 3 + jw * 3 + 2
									- kc;

							tmpacc.at<cv::Vec3f>(ih, jw)[kc] = tmpdata[idx];

							if (maxvalue < tmpdata[idx])
								maxvalue = tmpdata[idx];

						}
				tmpaccs.push_back(tmpacc);
			}
		}

		cout << "maxvalue\t" << maxvalue << endl;

		int idx = 0;
		for (int lh = 0; lh < lighthnum; lh++) {
			for (int lv = 0; lv < lightvnum; lv++) {

				bool hbor = (lh == 0) | (lh == lighthnum - 1);
				bool vbor = (lv == 0) | (lv == lightvnum - 1);
				if (!(hbor | vbor))
					continue;

				cv::Mat tmpacc = tmpaccs[idx];
				idx++;

				char name[256];

				/*
				 // Write to file!
				 sprintf(name, "%s/combine-light_%d_%d.txt", folder, lh, lv);
				 cv::FileStorage fs(name, cv::FileStorage::WRITE);
				 fs << "mat1" << tmpacc;
				 */

				sprintf(name, "%s/combine-light_%d_%d.png", folder, lh, lv);

				for (int ih = 0; ih < height; ih++)
					for (int jw = 0; jw < width; jw++)
						for (int kc = 0; kc < 3; kc++)
							tmpshow.at<cv::Vec3b>(ih, jw)[kc] = (uchar) (255.0f
									* tmpacc.at<cv::Vec3f>(ih, jw)[kc]
									/ maxvalue);

				cv::imwrite(name, tmpshow);
				// cv::imshow("tmp", tmpshow);
				// cv::waitKey(33);
			}
		}
	}
}

